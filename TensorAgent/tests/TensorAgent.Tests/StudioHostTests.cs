// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Net;
using System.Reflection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Chat;
using TensorSharp.Models;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.Server.Hosting;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Security.Cryptography;
using TensorAgent.Core.Hosting;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat.Studio;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Skills;
using TensorSharp.Studio;

namespace TensorAgent.Tests;

/// <summary>Real loopback transport and command persistence. No model inference is simulated as a product result.</summary>
public sealed class StudioHostTests : IAsyncLifetime
{
    readonly string directory = Path.Combine(Path.GetTempPath(), "tensorstudio-host-" + Guid.NewGuid().ToString("N"));
    StudioService studio = null!;
    public Task InitializeAsync()
    {
        Directory.CreateDirectory(directory);
        studio = new StudioService(Path.Combine(directory, "studio"), options: new StudioOptions { EnableExternalProcesses = false });
        return Task.CompletedTask;
    }
    public async Task DisposeAsync() { await studio.DisposeAsync(); Directory.Delete(directory, true); }

    [Theory]
    [InlineData("", "", false)]
    [InlineData("1", "https://evil.example", false)]
    [InlineData("1", "http://localhost:8001", false)]
    [InlineData("1", "http://localhost:8000", true)]
    [InlineData("1", "", true)]
    public void WritesRequireCustomHeaderAndMatchingOrigin(string marker, string origin, bool expected) =>
        Assert.Equal(expected, StudioHostBoundary.AllowsMutation(marker, origin, "http", "localhost:8000"));

    [Theory]
    [InlineData("127.0.0.1", "localhost", false, true)]
    [InlineData("127.0.0.1", "LOCALHOST", false, true)]
    [InlineData("127.0.0.1", "127.0.0.1", false, true)]
    [InlineData("::1", "[::1]", false, true)]
    [InlineData("::1", "::1", false, true)]
    [InlineData("::ffff:127.0.0.1", "[::ffff:127.0.0.1]", false, true)]
    [InlineData("127.0.0.1", "untrusted.example", false, false)]
    [InlineData("127.0.0.1", "localhost.evil.example", false, false)]
    [InlineData("127.0.0.1", "192.168.1.2", false, false)]
    [InlineData("192.168.1.2", "localhost", false, false)]
    [InlineData("::ffff:192.168.1.2", "localhost", false, false)]
    [InlineData("192.168.1.2", "studio.example", true, true)]
    [InlineData(null, "localhost", false, false)]
    public void StandaloneStudioRequiresLoopbackPeerAndHostUnlessOperatorOptsIn(string? peer, string host, bool allowRemote, bool expected) =>
        Assert.Equal(expected, StudioHostBoundary.AllowsServerConnection(peer == null ? null! : IPAddress.Parse(peer), host, allowRemote));

    [Theory]
    [InlineData("/uploads/../private.mp4")]
    [InlineData("/uploads/%2e%2e%2fprivate.mp4")]
    [InlineData("/uploads/%2fprivate.mp4")]
    [InlineData("https://remote.example/video.mp4")]
    [InlineData("/uploads/file.mp4?credential=secret")]
    public void ChatImportsCannotReadArbitraryPathsOrFetchRemoteUrls(string path)
    {
        var boundary = new StudioHostBoundary(studio, directory);
        Assert.Throws<StudioException>(() => boundary.ResolveChatUpload(path));
    }

    [Fact]
    public async Task AuthenticatedLoopbackRequiresHeaderAndPersistsProjectCommands()
    {
        string web = Path.Combine(directory, "web"); Directory.CreateDirectory(web);
        await File.WriteAllTextAsync(Path.Combine(web, "index.html"), "<html><body>Chat</body></html>");
        using var server = new LoopbackServer { StaticRoot = web };
        server.MapStudio(studio, directory); server.Start();
        using var client = new HttpClient(new HttpClientHandler { CookieContainer = new CookieContainer() });
        var create = new { action = "projects/create", payload = new { name = "Persisted studio" } };
        Assert.Equal(HttpStatusCode.Forbidden, (await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", create)).StatusCode);
        string html = await client.GetStringAsync(server.EntryUrl);
        Assert.Contains("/studio.js", html); Assert.Contains("/studio.css", html);
        Assert.Equal(HttpStatusCode.Forbidden, (await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", create)).StatusCode);
        client.DefaultRequestHeaders.Add("X-TensorSharp-Studio", "1");
        client.DefaultRequestHeaders.Add("Origin", "https://other.example");
        Assert.Equal(HttpStatusCode.Forbidden, (await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", create)).StatusCode);
        client.DefaultRequestHeaders.Remove("Origin");
        var response = await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", create);
        response.EnsureSuccessStatusCode();
        var project = (await response.Content.ReadFromJsonAsync<StudioProject>(StudioJson.Options))!;
        var edit = new { action = "commands/apply", payload = new { projectId = project.Id, expectedRevision = project.Revision, commandId = StudioJson.NewId(), commands = new[] { new { type = "addTrack", name = "Music", kind = "audio" } } } };
        (await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", edit)).EnsureSuccessStatusCode();
        Assert.Equal(3, (await studio.GetProjectAsync(project.Id)).Tracks.Count);
        Assert.Equal(HttpStatusCode.Conflict, (await client.PostAsJsonAsync(server.BaseUrl + "/api/studio", new { action = "commands/undo", payload = new { projectId = project.Id, expectedRevision = 0, commandId = StudioJson.NewId() } })).StatusCode);
        Assert.Equal("text/javascript", (await client.GetAsync(server.BaseUrl + "/studio.js")).Content.Headers.ContentType!.MediaType);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task DeleteAssetUsesSameRevisionSafeUndoableCommandInBothHosts(bool standaloneServer)
    {
        var project = await studio.CreateProjectAsync("Asset deletion");
        var otherProject = await studio.CreateProjectAsync("Isolated project");
        byte[] bytes = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOuoAAAAASUVORK5CYII=");
        var asset = new StudioAsset { Name = "Delete me.png", FileName = StudioJson.NewId() + ".png", Kind = "image", ContentType = "image/png", Width = 1, Height = 1, Bytes = bytes.Length, Sha256 = Convert.ToHexStringLower(SHA256.HashData(bytes)) };
        var retained = StudioJson.Clone(asset); retained.Id = StudioJson.NewId(); retained.FileName = StudioJson.NewId() + ".png"; retained.Name = "Keep me.png";
        // Seed deterministic media metadata: transport/edit tests require no decoder or loaded model.
        string projectDirectory = Path.Combine(directory, "studio", "projects", project.Id);
        string documentPath = Path.Combine(projectDirectory, "project.json");
        var document = JsonNode.Parse(await File.ReadAllTextAsync(documentPath))!;
        foreach (var item in new[] { asset, retained })
        {
            document["project"]!["assets"]!.AsArray().Add(JsonSerializer.SerializeToNode(item, StudioJson.Options));
            Directory.CreateDirectory(Path.Combine(projectDirectory, "assets"));
            await File.WriteAllBytesAsync(Path.Combine(projectDirectory, "assets", item.FileName), bytes);
        }
        await File.WriteAllTextAsync(documentPath, document.ToJsonString());
        string linkGroup = StudioJson.NewId();
        project = await studio.ApplyCommandsAsync(project.Id, project.Revision, StudioJson.NewId(), [
            new() { Type = "insertClip", AssetId = asset.Id, TrackId = project.Tracks[0].Id, StartFrame = 0, DurationFrames = 24, LinkGroup = linkGroup },
            new() { Type = "insertClip", AssetId = asset.Id, TrackId = project.Tracks[0].Id, StartFrame = 24, DurationFrames = 24 },
            new() { Type = "insertClip", AssetId = retained.Id, TrackId = project.Tracks[0].Id, StartFrame = 48, DurationFrames = 24, LinkGroup = linkGroup }
        ]);
        string mediaPath = studio.GetAssetPath(project.Id, asset.Id);
        await using var webHost = standaloneServer ? BuildStudioServer() : null;
        using var agentHost = standaloneServer ? null : new LoopbackServer();
        string baseUrl;
        if (webHost is not null) { await webHost.StartAsync(); baseUrl = webHost.Urls.Single(); }
        else { agentHost!.MapStudio(studio, directory); agentHost.Start(); baseUrl = agentHost.BaseUrl; }
        using var client = new HttpClient(new HttpClientHandler { CookieContainer = new CookieContainer() });
        if (agentHost is not null) await client.GetAsync(agentHost.EntryUrl);
        string api = baseUrl + "/api/studio";
        string media = api + $"/projects/{project.Id}/assets/{asset.Id}";
        object DeleteRequest(string projectId, long revision, bool? removeClips = null, string? commandId = null)
        {
            var command = new Dictionary<string, object> { ["type"] = "deleteAsset", ["assetId"] = asset.Id };
            if (removeClips.HasValue) command["removeTimelineClips"] = removeClips.Value;
            return new { action = "commands/apply", payload = new { projectId, expectedRevision = revision, commandId = commandId ?? StudioJson.NewId(), commands = new[] { command } } };
        }
        async Task<StudioProject> Apply(object request)
        {
            using var response = await client.PostAsJsonAsync(api, request);
            response.EnsureSuccessStatusCode();
            return (await response.Content.ReadFromJsonAsync<StudioProject>(StudioJson.Options))!;
        }
        Assert.Equal(HttpStatusCode.Forbidden, (await client.PostAsJsonAsync(api, DeleteRequest(project.Id, project.Revision, true))).StatusCode);
        client.DefaultRequestHeaders.Add("X-TensorSharp-Studio", "1");
        using (var blocked = await client.PostAsJsonAsync(api, DeleteRequest(project.Id, project.Revision)))
        {
            Assert.Equal(HttpStatusCode.BadRequest, blocked.StatusCode);
            Assert.Equal("asset_in_use", (await blocked.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("code").GetString());
        }
        Assert.Equal(project.Revision, (await studio.GetProjectAsync(project.Id)).Revision);
        Assert.Equal(HttpStatusCode.NotFound, (await client.PostAsJsonAsync(api, DeleteRequest(otherProject.Id, otherProject.Revision, true))).StatusCode);
        Assert.Equal(HttpStatusCode.Conflict, (await client.PostAsJsonAsync(api, DeleteRequest(project.Id, project.Revision - 1, true))).StatusCode);
        Assert.Equal(bytes, await client.GetByteArrayAsync(media));

        var confirmed = DeleteRequest(project.Id, project.Revision, true, StudioJson.NewId());
        var deleted = await Apply(confirmed);
        Assert.Equal(project.Revision + 1, deleted.Revision);
        Assert.Equal(retained.Id, Assert.Single(deleted.Assets).Id);
        Assert.Equal(retained.Id, Assert.Single(deleted.Clips).AssetId); // Linked clips for other assets survive.
        Assert.Equal(deleted.Revision, (await Apply(confirmed)).Revision); // A retried confirmation is idempotent.
        Assert.Equal(HttpStatusCode.NotFound, (await client.GetAsync(media)).StatusCode);
        Assert.Equal(bytes, await File.ReadAllBytesAsync(mediaPath)); // Undo/captured jobs retain the original bytes.

        var undone = await Apply(new { action = "commands/undo", payload = new { projectId = project.Id, expectedRevision = deleted.Revision, commandId = StudioJson.NewId() } });
        Assert.Equal(2, undone.Assets.Count); Assert.Equal(3, undone.Clips.Count);
        Assert.Equal(bytes, await client.GetByteArrayAsync(media));
        var redone = await Apply(new { action = "commands/redo", payload = new { projectId = project.Id, expectedRevision = undone.Revision, commandId = StudioJson.NewId() } });
        Assert.Equal(retained.Id, Assert.Single(redone.Assets).Id);
        Assert.Equal(retained.Id, Assert.Single(redone.Clips).AssetId);
        Assert.Equal(bytes, await File.ReadAllBytesAsync(mediaPath));
        Assert.Equal(otherProject.Revision, (await studio.GetProjectAsync(otherProject.Id)).Revision);
    }

    WebApplication BuildStudioServer()
    {
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Testing" });
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Logging.ClearProviders();
        builder.Services.AddSingleton(new StudioHostBoundary(studio, directory));
        var app = builder.Build();
        app.MapVideoStudio();
        return app;
    }

    [Fact]
    public async Task AgentUsesSameProposalCommandsAndCannotOverwriteNewManualRevision()
    {
        var project = await studio.CreateProjectAsync("Agent scope");
        using var model = new TextModelService();
        var scope = StudioChatScope.Create(studio, model, JsonSerializer.SerializeToElement(new
        { studio = new { projectId = project.Id, expectedRevision = project.Revision, selectedClipIds = Array.Empty<string>() } }), directory);
        var context = new SkillToolContext(Array.Empty<Skill>()) { Studio = scope };
        var malformed = SkillTools.Execute(new ToolCall { Name = "studio_propose", Arguments = new()
        { ["summary"] = "Invalid shape", ["commands"] = "[{\"addTrack\":{\"kind\":\"audio\",\"name\":\"Music\"}}]" } }, context);
        Assert.False(malformed.Ok); Assert.Contains("FLAT object", malformed.Content); Assert.Contains("without asking", malformed.Content);
        Assert.Empty(await studio.ListProposalsAsync(project.Id));
        var proposal = SkillTools.Execute(new ToolCall { Name = "studio_propose", Arguments = new()
        { ["summary"] = "Add an audio track", ["commands"] = "[{\"type\":\"addTrack\",\"kind\":\"audio\",\"name\":\"Music\"}]" } }, context);
        Assert.True(proposal.Ok, proposal.Content);
        Assert.Equal(2, (await studio.GetProjectAsync(project.Id)).Tracks.Count); // Proposal is not an approval.
        var saved = JsonSerializer.Deserialize<StudioProposal>(proposal.Content, StudioJson.Options)!;
        await studio.ApplyProposalAsync(project.Id, saved.Id, StudioJson.NewId());
        Assert.Equal(3, (await studio.GetProjectAsync(project.Id)).Tracks.Count);
        var stale = SkillTools.Execute(new ToolCall { Name = "studio_propose", Arguments = new()
        { ["summary"] = "Stale edit", ["commands"] = "[{\"type\":\"addTrack\",\"kind\":\"audio\",\"name\":\"Old\"}]" } }, context);
        Assert.False(stale.Ok); Assert.Contains("revision", stale.Content, StringComparison.OrdinalIgnoreCase);
        await studio.UndoAsync(project.Id, 1, StudioJson.NewId());
        Assert.Equal(2, (await studio.GetProjectAsync(project.Id)).Tracks.Count);
    }

    [Fact]
    public async Task RelinkTransportRequiresHeaderAndExactOriginalBytesAndReturnsCamelCase()
    {
        var project = await studio.CreateProjectAsync("Missing media");
        byte[] original = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOuoAAAAASUVORK5CYII=");
        var asset = new StudioAsset { Name = "Original.png", FileName = StudioJson.NewId() + ".png", Kind = "image", ContentType = "image/png", Width = 1, Height = 1, Bytes = original.Length, Sha256 = Convert.ToHexStringLower(SHA256.HashData(original)) };
        // Seed persisted metadata with its deliberately missing original file to exercise recovery without FFmpeg.
        string documentPath = Path.Combine(directory, "studio", "projects", project.Id, "project.json");
        var document = JsonNode.Parse(await File.ReadAllTextAsync(documentPath))!;
        document["project"]!["assets"]!.AsArray().Add(JsonSerializer.SerializeToNode(asset, StudioJson.Options));
        await File.WriteAllTextAsync(documentPath, document.ToJsonString());
        using var server = new LoopbackServer(); server.MapStudio(studio, directory); server.Start();
        using var client = new HttpClient(new HttpClientHandler { CookieContainer = new CookieContainer() });
        await client.GetAsync(server.EntryUrl);
        string url = server.BaseUrl + $"/api/studio/projects/{project.Id}/assets/{asset.Id}/relink";
        Assert.Equal(HttpStatusCode.Forbidden, (await client.PostAsync(url, new ByteArrayContent(original))).StatusCode);
        client.DefaultRequestHeaders.Add("X-TensorSharp-Studio", "1");
        byte[] wrong = original.ToArray(); wrong[20] ^= 1;
        Assert.Equal(HttpStatusCode.BadRequest, (await client.PostAsync(url, new ByteArrayContent(wrong))).StatusCode);
        Assert.True((await studio.GetProjectAsync(project.Id)).Assets[0].Missing);
        var response = await client.PostAsync(url, new ByteArrayContent(original)); response.EnsureSuccessStatusCode();
        using var saved = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal(project.Id, saved.RootElement.GetProperty("id").GetString());
        Assert.False(saved.RootElement.GetProperty("assets")[0].GetProperty("missing").GetBoolean());
        Assert.Equal(original, await File.ReadAllBytesAsync(studio.GetAssetPath(project.Id, asset.Id)));
        Assert.Equal(project.Revision, (await studio.GetProjectAsync(project.Id)).Revision);
    }

    [Fact]
    public async Task PlannedGenerationPersistsWithoutPretendingToGenerateOrSwitchModels()
    {
        var project = await studio.CreateProjectAsync("Shot plan");
        using var model = new TextModelService();
        var scope = StudioChatScope.Create(studio, model, JsonSerializer.SerializeToElement(new { studio = new { projectId = project.Id, expectedRevision = 0 } }), directory);
        var result = SkillTools.Execute(new ToolCall { Name = "studio_generate", Arguments = new()
        { ["summary"] = "Opening shot", ["request"] = "{\"prompt\":\"A sunrise over mountains\",\"width\":512,\"height\":512,\"frameCount\":17,\"fps\":24}" } }, new SkillToolContext(Array.Empty<Skill>()) { Studio = scope });
        Assert.True(result.Ok, result.Content); Assert.Contains("awaiting_user_start", result.Content);
        var plans = await studio.ListGenerationDraftsAsync(project.Id);
        Assert.Single(plans); Assert.Null(plans[0].JobId); Assert.Empty((await studio.GetProjectAsync(project.Id)).Assets);
        var error = await Assert.ThrowsAsync<StudioException>(() => studio.StartGenerationDraftAsync(project.Id, plans[0].Id, 0, "", CancellationToken.None));
        Assert.Equal("model_required", error.Code);
    }

    [Fact]
    public void StudioToolsWithoutExplicitProjectAreRejected()
    {
        var result = SkillTools.Execute(new ToolCall { Name = "studio_render", Arguments = new() }, new SkillToolContext(Array.Empty<Skill>()));
        Assert.False(result.Ok); Assert.Contains("explicitly selected", result.Content);
    }

    [Fact]
    public void PlanningWorkflowInstructionsRequireAnExplicitStudioScope()
    {
        var registry = new SkillRegistry(new SkillRegistryOptions());
        var options = OptionsFor("");
        var scoped = SkillRequestPlan.Create(registry, Array.Empty<string>(), false, new(), "qwen35", 32768, options, out var unknown,
            studio: new StudioToolScope((_, _) => new SkillToolResult(true, "{}", null, null)));
        Assert.Empty(unknown); Assert.NotNull(scoped);
        Assert.Contains("call studio_inspect first", scoped.Prompt.Instructions);
        Assert.Contains("never ask the user to supply IDs", scoped.Prompt.Instructions);
        Assert.Contains("call studio_propose", scoped.Prompt.Instructions);
        var ordinary = SkillRequestPlan.Create(registry, Array.Empty<string>(), false, new(), "qwen35", 32768, options, out _);
        Assert.DoesNotContain("video_studio_workflow", ordinary?.Prompt.Instructions ?? "");
    }

    [Fact]
    public async Task StudioChatOnVideoModelCannotBecomeAnImplicitVideoPrompt()
    {
        string path = Path.Combine(directory, "test-model.gguf");
        using (var writer = new BinaryWriter(File.Create(path)))
        { writer.Write(0x46554747u); writer.Write(3u); writer.Write(0UL); writer.Write(0UL); writer.Write(new byte[8]); }
        using var video = new RefusingVideoModel(path);
        using var models = new ModelService();
        FieldInfo field = models.LifecycleService.GetType().GetField("_model", BindingFlags.Instance | BindingFlags.NonPublic)!;
        field.SetValue(models.LifecycleService, video);
        try
        {
            var options = OptionsFor(path);
            var chat = new WebUiChatService(models, new SessionManager(), options, new UploadStoragePolicy(directory),
                new SkillRegistry(new SkillRegistryOptions()), codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
            var body = JsonSerializer.SerializeToElement(new { messages = new[] { new { role = "user", content = "Trim my selected clip" } }, studio = new { projectId = StudioJson.NewId(), expectedRevision = 0 } });
            var error = await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () =>
            {
                await foreach (var frame in ImageTurns.FramesFor(chat, body, CancellationToken.None, null, ImageTurns.Planner.For(chat, directory))) { }
            });
            Assert.Contains("text planning model", error.Message); Assert.Equal(0, video.Calls);
        }
        finally { field.SetValue(models.LifecycleService, null); }
    }

    ServerHostingOptions OptionsFor(string path) => new(startupModelPath: path, startupMmProjPath: null, defaultBackend: "cpu",
                supportedBackends: new[] { new BackendOption("cpu", "CPU") }, defaultMaxTokens: 128, maxTokensPinned: false,
                defaultVideoFrames: 17, defaultVideoFps: 24, defaultVideoWidth: 64, defaultVideoHeight: 64,
                defaultVideoSteps: 8, defaultVideoMode: "t2v", uploadDirectory: directory, logDirectory: directory,
                fileLoggingEnabled: false, samplingDefaults: null);

    sealed class RefusingVideoModel(string path) : ModelBase(path, BackendType.Cpu), IVideoGenerationModel
    {
        public int Calls { get; private set; }
        public string VideoModelFamily => "ltx-2.5";
        public bool SupportsAudio => false;
        public bool SupportsImageConditioning => false;
        public bool SupportsEndImageConditioning => false;
        public bool SupportsReferenceConditioning => false;
        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams? p = null) { ++Calls; throw new InvalidOperationException("Must never generate for an editing request."); }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    sealed class TextModelService : ModelService { public override string Architecture => "qwen35"; }
}
