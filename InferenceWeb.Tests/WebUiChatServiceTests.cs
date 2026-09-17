// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="WebUiChatService"/> without a model: the preflight refusals and the
/// transport-free replies must match what <see cref="WebUiAdapter"/> answered when
/// the code lived inside it, because the Web UI page — served by the Server and by
/// the iOS app's loopback server alike — is written against exactly those shapes.
/// </summary>
public class WebUiChatServiceTests : IDisposable
{
    private readonly string _baseDir;

    public WebUiChatServiceTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-webui-service-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    [Fact]
    public void LegacyPublicConstructorSignatureRemainsAvailableForCompiledConsumers()
    {
        Type[] legacyParameters =
        {
            typeof(ModelService),
            typeof(SessionManager),
            typeof(ServerHostingOptions),
            typeof(UploadStoragePolicy),
            typeof(SkillRegistry),
            typeof(ICodeRunner),
            typeof(SessionWorkspaceManager),
            typeof(TensorSharp.AgentHost.CodeExec.CodeArtifactStore),
            typeof(Microsoft.Extensions.Logging.ILoggerFactory),
            typeof(string),
        };

        Assert.NotNull(typeof(WebUiChatService).GetConstructor(legacyParameters));
    }

    private ServerHostingOptions Options(
        bool skillsAllowScripts = false,
        bool skillsAllowNetwork = false,
        SkillSandboxMode skillsSandbox = SkillSandboxMode.Required) => new(
        startupModelPath: Path.Combine(_baseDir, "models", "foo.gguf"),
        startupMmProjPath: Path.Combine(_baseDir, "models", "mmproj-foo.gguf"),
        defaultBackend: "ggml_cpu",
        supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
        defaultMaxTokens: 100,
        maxTokensPinned: false,
        defaultVideoFrames: 0,
        defaultVideoFps: 0,
        defaultVideoWidth: 0,
        defaultVideoHeight: 0,
        defaultVideoSteps: 0,
        defaultVideoMode: null,
        uploadDirectory: _baseDir,
        logDirectory: Path.Combine(_baseDir, "logs"),
        fileLoggingEnabled: false,
        samplingDefaults: null,
        skillsAllowScripts: skillsAllowScripts,
        skillsSandbox: skillsSandbox,
        skillsAllowNetwork: skillsAllowNetwork);

    private sealed record Fixture(
        WebUiChatService Service,
        ModelService Model,
        SessionManager Sessions,
        UploadStoragePolicy Uploads,
        ServerHostingOptions Options,
        SkillRegistry Skills);

    private Fixture Build(UploadStoragePolicy uploads = null, ModelService model = null)
    {
        model ??= new ModelService();
        var sessions = new SessionManager();
        var options = Options();
        var skills = new SkillRegistry(new SkillRegistryOptions());
        uploads ??= new UploadStoragePolicy(_baseDir);
        var service = new WebUiChatService(
            model, sessions, options, uploads, skills,
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);
        return new Fixture(service, model, sessions, uploads, options, skills);
    }

    /// <summary>Reads one property out of an anonymous payload object as text.</summary>
    private static string? Field(object payload, string name)
    {
        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(payload));
        JsonElement value = doc.RootElement.GetProperty(name);
        return value.ValueKind == JsonValueKind.String ? value.GetString() : value.ToString();
    }

    private static JsonElement Json(string json) => JsonSerializer.Deserialize<JsonElement>(json);

    private static async Task<WebUiRequestRejectedException> RejectionOf(IAsyncEnumerable<object> frames)
    {
        return await Assert.ThrowsAsync<WebUiRequestRejectedException>(async () =>
        {
            await foreach (object _ in frames) { }
        });
    }

    private sealed class ActionDisposable(Action dispose) : IDisposable
    {
        private Action? _dispose = dispose;
        public void Dispose() => Interlocked.Exchange(ref _dispose, null)?.Invoke();
    }

    [Fact]
    public void APreParsedSkillLoopAnswerCountsAsVisibleContent()
    {
        Assert.True(WebUiChatService.HasParsedAnswerContent(
            ChatStreamUpdate.Parsed("The answer.", null, null)));
        Assert.False(WebUiChatService.HasParsedAnswerContent(
            ChatStreamUpdate.Parsed(string.Empty, "thinking", null)));
        Assert.False(WebUiChatService.HasParsedAnswerContent(
            ChatStreamUpdate.ToolProgress("running", "shell", "output")));
    }

    [Fact]
    public void HostSkillRoutingRespectsEveryExplicitOptOutAndCallerTools()
    {
        Assert.True(WebUiChatService.MayInferSkillRoute(
            skillsEnabled: true, requestedDiscovery: null, clientTools: null));
        Assert.False(WebUiChatService.MayInferSkillRoute(
            skillsEnabled: false, requestedDiscovery: null, clientTools: null));
        Assert.False(WebUiChatService.MayInferSkillRoute(
            skillsEnabled: true, requestedDiscovery: false, clientTools: null));
        Assert.False(WebUiChatService.MayInferSkillRoute(
            skillsEnabled: true,
            requestedDiscovery: null,
            clientTools: new[] { new ToolFunction { Name = "caller's_tool" } }));
    }

    [Fact]
    public void RoutedWorkflowPreflightReportsEveryDisabledRequiredCapability()
    {
        WebUiSkillRoute route = RoutedWorkflow(requiresNetwork: true);
        ServerHostingOptions options = Options(
            skillsAllowScripts: false, skillsAllowNetwork: false);

        string error = WebUiChatService.RoutedWorkflowPreflightError(
            route, plan: null, options);

        Assert.Contains("skills_run tool is disabled", error, StringComparison.Ordinal);
        Assert.Contains("network access for skill scripts is disabled", error, StringComparison.Ordinal);
        Assert.Contains("Then retry", error, StringComparison.Ordinal);
        Assert.False(options.SkillsAllowScripts);
        Assert.False(options.SkillsAllowNetwork);
    }

    [Fact]
    public void RoutedWorkflowPreflightRejectsAMissingSkillsRunChannelEvenWhenPermissionsAreEnabled()
    {
        string error = WebUiChatService.RoutedWorkflowPreflightError(
            RoutedWorkflow(requiresNetwork: true),
            plan: null,
            Options(
                skillsAllowScripts: true,
                skillsAllowNetwork: true,
                skillsSandbox: SkillSandboxMode.Off));

        Assert.Contains("skills_run tool is unavailable", error, StringComparison.Ordinal);
        Assert.Contains("tool-capable chat model", error, StringComparison.Ordinal);
    }

    [Fact]
    public void RoutedWorkflowPreflightAcceptsAnAvailableSkillsRunChannelAndExplicitNetworkPermission()
    {
        string skillsRoot = Path.Combine(_baseDir, "preflight-skills");
        string skillRoot = Path.Combine(skillsRoot, "research");
        Directory.CreateDirectory(skillRoot);
        File.WriteAllText(Path.Combine(skillRoot, SkillManifestParser.SkillFileName), """
            ---
            name: research
            description: Researches the web.
            ---
            # Research
            """);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skillsRoot } });
        ServerHostingOptions options = Options(
            skillsAllowScripts: true,
            skillsAllowNetwork: true,
            skillsSandbox: SkillSandboxMode.Off);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry,
            new[] { "research" },
            discovery: false,
            clientTools: null,
            architecture: "nemotron_h_moe",
            contextTokens: 8192,
            options,
            out IReadOnlyList<string> unknown);

        Assert.Empty(unknown);
        Assert.NotNull(plan);
        Assert.Null(WebUiChatService.RoutedWorkflowPreflightError(
            RoutedWorkflow(requiresNetwork: true), plan, options));
    }

    [Fact]
    public async Task InferredRoutedWorkflowRejectsMissingCapabilitiesBeforeGeneration()
    {
        string modelPath = WriteMinimalGguf("routed-preflight.gguf");
        using var model = new ModelService(
            NullLogger<ModelService>.Instance,
            (path, _, _, _) => new ContextReportingModel(path, declaredContext: 8192, activeContext: 8192));
        model.LoadModel(modelPath, null, "cpu");

        string skillsRoot = Path.Combine(_baseDir, "routed-preflight-skills");
        string skillRoot = Path.Combine(skillsRoot, "research");
        Directory.CreateDirectory(skillRoot);
        File.WriteAllText(Path.Combine(skillRoot, SkillManifestParser.SkillFileName), """
            ---
            name: research
            description: Researches the web.
            ---
            # Research
            """);
        var skills = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skillsRoot } });
        var sessions = new SessionManager();
        ServerHostingOptions options = Options(
            skillsAllowScripts: false, skillsAllowNetwork: true);
        var service = new WebUiChatService(
            model,
            sessions,
            options,
            new UploadStoragePolicy(_baseDir),
            skills,
            codeRunner: null,
            workspaces: null,
            codeArtifacts: null,
            NullLoggerFactory.Instance,
            WebUiChatService.DefaultArtifactUriPrefix,
            skillRouter: (_, _, _) => RoutedWorkflow(requiresNetwork: true));
        bool generationHookFired = false;
        service.OnChatRequest = (_, _) => generationHookFired = true;

        WebUiRequestRejectedException ex = await RejectionOf(service.ChatStreamAsync(
            Json("""{"messages":[{"role":"user","content":"make a researched deck"}]}"""),
            CancellationToken.None));

        Assert.Equal(503, ex.StatusCode);
        Assert.Equal("routed_workflow_unavailable", Field(ex.Payload, "code"));
        Assert.Contains("skills_run tool is disabled", ex.Message, StringComparison.Ordinal);
        Assert.False(generationHookFired, "preflight rejection must happen before generation is accepted");
        Assert.True(options.SkillsAllowNetwork);
    }

    /// <summary>
    /// TensorAgent opens a new session every time a saved chat is opened. Scoped by
    /// session alone, each reopen could not continue that chat's own cached state
    /// (another session's scope) and re-prefilled everything past the system prompt.
    /// Sessions bound to one conversation share its cache scope; others do not.
    /// </summary>
    [Fact]
    public void SessionsBoundToOneConversation_ShareItsCacheScope()
    {
        using var model = new ModelService(NullLogger<ModelService>.Instance);
        var sessions = new SessionManager();
        var service = new WebUiChatService(
            model, sessions, Options(), new UploadStoragePolicy(_baseDir),
            new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance, WebUiChatService.DefaultArtifactUriPrefix);

        ChatSession first = sessions.CreateSession();
        ChatSession reopened = sessions.CreateSession();
        ChatSession other = sessions.CreateSession();
        Assert.NotEqual(first.ResolveCacheScope(null), reopened.ResolveCacheScope(null));

        Assert.True(service.BindSessionConversation(first.Id, "conv-1"));
        Assert.True(service.BindSessionConversation(reopened.Id, "conv-1"));
        Assert.True(service.BindSessionConversation(other.Id, "conv-2"));
        Assert.Equal(first.ResolveCacheScope(null), reopened.ResolveCacheScope(null));
        Assert.NotEqual(first.ResolveCacheScope(null), other.ResolveCacheScope(null));
        // The key never appears in the scope, and a new chat in the session still starts a new one.
        Assert.DoesNotContain("conv-1", first.ResolveCacheScope(null), StringComparison.Ordinal);
        lock (reopened.HistoryLock) reopened.ResetConversation();
        Assert.NotEqual(first.ResolveCacheScope(null), reopened.ResolveCacheScope(null));

        // The shared default session and unknown ids are never bound.
        Assert.False(service.BindSessionConversation(SessionManager.DefaultSessionId, "conv-1"));
        Assert.False(service.BindSessionConversation("no-such-session", "conv-1"));
    }

    [Fact]
    public async Task RoutedArtifactSetupFailureAlsoCarriesTheBrowserRollbackCode()
    {
        string modelPath = WriteMinimalGguf("routed-artifact-preflight.gguf");
        using var model = new ModelService(
            NullLogger<ModelService>.Instance,
            (path, _, _, _) => new ContextReportingModel(path, declaredContext: 8192, activeContext: 8192));
        model.LoadModel(modelPath, null, "cpu");

        var service = new WebUiChatService(
            model,
            new SessionManager(),
            Options(),
            new UploadStoragePolicy(_baseDir),
            new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null,
            workspaces: null,
            codeArtifacts: null,
            NullLoggerFactory.Instance,
            WebUiChatService.DefaultArtifactUriPrefix,
            skillRouter: (_, _, _) => new WebUiSkillRoute(
                Array.Empty<string>(),
                "Create the deliverable.",
                new WebUiArtifactRequirement(
                    ".pptx", Array.Empty<WebUiSkillRunRequirement>()),
                RequiresNetwork: false));
        bool generationHookFired = false;
        service.OnChatRequest = (_, _) => generationHookFired = true;

        WebUiRequestRejectedException ex = await RejectionOf(service.ChatStreamAsync(
            Json("""{"messages":[{"role":"user","content":"make a deck"}]}"""),
            CancellationToken.None));

        Assert.Equal(503, ex.StatusCode);
        Assert.Equal("routed_workflow_unavailable", Field(ex.Payload, "code"));
        Assert.Contains("durable artifact downloads", ex.Message, StringComparison.Ordinal);
        Assert.False(generationHookFired);
    }

    private static WebUiSkillRoute RoutedWorkflow(bool requiresNetwork) => new(
        new[] { "research" },
        "Use research.",
        new WebUiArtifactRequirement(
            ".pptx",
            new[] { new WebUiSkillRunRequirement("research", "scripts/research.py") }),
        RequiresNetwork: requiresNetwork);

    // ---- /api/chat preflight -------------------------------------------------

    [Fact]
    public async Task ChatStream_WithNoModelLoaded_Rejects400BeforeTheFirstFrame()
    {
        Fixture f = Build();
        bool hookFired = false;
        f.Service.OnChatRequest = (_, _) => hookFired = true;

        var ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json("""{"messages":[{"role":"user","content":"hi"}]}"""), CancellationToken.None));

        Assert.Equal(400, ex.StatusCode);
        Assert.Equal("No model loaded", ex.Message);
        Assert.Equal("""{"error":"No model loaded"}""", JsonSerializer.Serialize(ex.Payload));
        Assert.False(hookFired, "OnChatRequest must not fire for a rejected request");
    }

    [Fact]
    public async Task ChatStream_RequestLeaseSpansPreflightAndIsReleasedOnRefusal()
    {
        Fixture f = Build();
        int acquired = 0;
        int released = 0;
        f.Service.AcquireChatRequestLease = _ =>
        {
            Interlocked.Increment(ref acquired);
            return new ActionDisposable(() => Interlocked.Increment(ref released));
        };

        WebUiRequestRejectedException ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json("""{"messages":[{"role":"user","content":"hi"}]}"""), CancellationToken.None));

        Assert.Equal(400, ex.StatusCode);
        Assert.Equal(1, acquired);
        Assert.Equal(1, released);
    }

    [Fact]
    public async Task ChatStream_RequestLeaseCanRejectBeforeAttachmentPreflight()
    {
        Fixture f = Build();
        f.Service.AcquireChatRequestLease = _ => throw new WebUiRequestRejectedException(
            409, new { code = "shared_draft_unavailable", error = "draft already discarded" });

        WebUiRequestRejectedException ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json("""{"messages":[{"role":"user","content":"hi"}]}"""), CancellationToken.None));

        Assert.Equal(409, ex.StatusCode);
        Assert.Equal("shared_draft_unavailable", Field(ex.Payload, "code"));
    }

    [Fact]
    public async Task ChatStream_ModelNamedInTheBody_IsRefusedBeforeAnythingElse()
    {
        Fixture f = Build();
        Assert.False(WebUiChatPolicy.TryValidateChatRequest("other.gguf", null, out string expected));

        var ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json("""{"model":"other.gguf","messages":[]}"""), CancellationToken.None));

        Assert.Equal(400, ex.StatusCode);
        Assert.Equal(expected, ex.Message);
    }

    [Fact]
    public async Task ChatStream_UnknownSession_Is404()
    {
        Fixture f = Build();

        var ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json("""{"sessionId":"deadbeef","messages":[]}"""), CancellationToken.None));

        Assert.Equal(404, ex.StatusCode);
        // Compare the payload's VALUES, not its serialized text: System.Text.Json escapes
        // an apostrophe as \u0027, so a literal comparison would assert the escaping rule
        // rather than the contract.
        Assert.Equal("Session 'deadbeef' not found or has been disposed.", Field(ex.Payload, "error"));
    }

    [Fact]
    public async Task ChatStream_NewChatOnAKnownSession_ResetsItBeforeTheNoModelRefusal()
    {
        Fixture f = Build();
        ChatSession session = f.Sessions.CreateSession();
        TranscriptTestHelper.RecordTurn(session, "earlier");

        var ex = await RejectionOf(f.Service.ChatStreamAsync(
            Json($$"""{"sessionId":"{{session.Id}}","newChat":true,"messages":[]}"""), CancellationToken.None));

        // The refusal order is the adapter's: session lookup and reset come before the
        // model check, so a New Chat on a model-less server still clears the desk.
        Assert.Equal(400, ex.StatusCode);
        Assert.Equal(0, session.TrackedTurnCount);
    }

    // ---- sessions and models ---------------------------------------------------

    [Fact]
    public void GetModels_MatchesTheAdapterByteForByte()
    {
        Fixture f = Build();
        var adapter = new WebUiAdapter(
            f.Model, new InferenceQueue(), f.Sessions, f.Options, f.Uploads, f.Skills,
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);

        var result = adapter.GetModels();
        object adapterValue = result.GetType().GetProperty("Value")?.GetValue(result);
        string viaAdapter = JsonSerializer.Serialize(adapterValue);
        string viaService = JsonSerializer.Serialize(f.Service.GetModels());

        Assert.Equal(viaService, viaAdapter);
        using var doc = JsonDocument.Parse(viaService);
        JsonElement root = doc.RootElement;
        Assert.Equal("foo.gguf", root.GetProperty("models")[0].GetString());
        Assert.Equal("mmproj-foo.gguf", root.GetProperty("mmProjModels")[0].GetString());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("loaded").ValueKind);
        Assert.False(root.GetProperty("visionReady").GetBoolean());
        Assert.False(root.GetProperty("acceptsVisionProjector").GetBoolean());
        Assert.Equal("ggml_cpu", root.GetProperty("defaultBackend").GetString());
        Assert.Equal("ggml_cpu", root.GetProperty("supportedBackends")[0].GetProperty("Value").GetString());
        Assert.Equal(0, root.GetProperty("contextTokens").GetInt32());
        Assert.Equal(0, root.GetProperty("modelContextTokens").GetInt32());
        Assert.Equal(100, root.GetProperty("defaultMaxTokens").GetInt32());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("video").ValueKind);
        Assert.True(root.GetProperty("skills").GetProperty("enabled").GetBoolean());
        Assert.DoesNotContain(_baseDir, viaService);
    }

    [Fact]
    public void GetModels_ReportsTheLoadedModelsDeclaredAndActiveContexts()
    {
        string modelPath = WriteMinimalGguf("context-reporting.gguf");
        using var model = new ModelService(
            NullLogger<ModelService>.Instance,
            (path, _, _, _) => new ContextReportingModel(path, declaredContext: 262144, activeContext: 32768));
        model.LoadModel(modelPath, null, "cpu");
        Fixture f = Build(model: model);

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(f.Service.GetModels()));
        JsonElement root = doc.RootElement;

        Assert.Equal(262144, root.GetProperty("modelContextTokens").GetInt32());
        Assert.Equal(32768, root.GetProperty("contextTokens").GetInt32());
    }

    [Fact]
    public void QueueStatus_IsIdleBeforeAnyRequest_AndCarriesTheLegacyCount()
    {
        Fixture f = Build();

        Assert.Equal(
            """{"busy":false,"processing":0,"pending_requests":0,"total_processed":7}""",
            JsonSerializer.Serialize(f.Service.GetQueueStatus(7)));
    }

    [Fact]
    public async Task Sessions_CreateThenDispose_RoundTripsThroughTheContract()
    {
        Fixture f = Build();

        using var created = JsonDocument.Parse(JsonSerializer.Serialize(f.Service.CreateSession()));
        string id = created.RootElement.GetProperty("sessionId").GetString();
        Assert.Equal(32, id.Length);
        Assert.True(DateTime.TryParse(created.RootElement.GetProperty("createdAt").GetString(), out _));
        Assert.NotNull(f.Sessions.GetSession(id));

        string disposed = JsonSerializer.Serialize(await f.Service.DisposeSessionAsync(id, CancellationToken.None));
        Assert.Equal($$"""{"ok":true,"sessionId":"{{id}}"}""", disposed);
        Assert.Null(f.Sessions.TryRemove(id));

        var again = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => f.Service.DisposeSessionAsync(id, CancellationToken.None));
        Assert.Equal(404, again.StatusCode);
        Assert.False(bool.Parse(Field(again.Payload, "ok")!));
        Assert.Equal($"Session '{id}' not found.", Field(again.Payload, "error"));

        var theDefault = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => f.Service.DisposeSessionAsync(SessionManager.DefaultSessionId, CancellationToken.None));
        Assert.Equal(400, theDefault.StatusCode);
        Assert.False(bool.Parse(Field(theDefault.Payload, "ok")!));
        Assert.Equal("Cannot dispose the default session.", Field(theDefault.Payload, "error"));
    }

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_baseDir, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);
        writer.Write(0UL); // tensors
        writer.Write(0UL); // metadata entries
        writer.Write(new byte[8]); // 32-byte data alignment
        return path;
    }

    private sealed class ContextReportingModel : ModelBase
    {
        public ContextReportingModel(string path, int declaredContext, int activeContext, string architecture = "qwen35")
            : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig
            {
                Architecture = architecture,
                DeclaredContextLength = declaredContext,
            };
            _maxContextLength = activeContext;
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
    }

    [Theory]
    [InlineData("deepseek41", false)]
    [InlineData("deepseek41", true)]
    [InlineData("nemotron_h_moe", false)]
    [InlineData("nemotron_h_moe", true)]
    [InlineData("nemotron_h", false)]
    public async Task ArchitecturesWithoutAnAudioTower_RejectAudioBeforeStreamOrGenerationHook(string architecture, bool withImage)
    {
        string modelPath = WriteMinimalGguf("audio-refusal.gguf");
        using var model = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new ContextReportingModel(path, 8192, 8192, architecture));
        model.LoadModel(modelPath, null, "cpu");
        Fixture f = Build(model: model);
        bool accepted = false;
        f.Service.OnChatRequest = (_, _) => accepted = true;
        string image = withImage ? ",\"imagePaths\":[\"photo.png\"]" : "";
        var error = await RejectionOf(f.Service.ChatStreamAsync(Json(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Describe this\",\"audioPaths\":[\"clip.wav\"]" + image + "}]}"),
            CancellationToken.None));
        Assert.Equal(400, error.StatusCode);
        Assert.Contains("does not support audio input", Field(error.Payload, "error"));
        Assert.False(accepted);

        // Direct service callers bypass HTTP validation but must still fail
        // before the fake model's unavailable engine or any prompt rendering.
        var history = new List<ChatMessage> { new() { Role = "user", Content = "Describe this",
            AudioPaths = new() { "clip.wav" }, ImagePaths = withImage ? new() { "photo.png" } : null } };
        var pipelineError = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (var _ in model.ChatStreamWithMetricsAsync(history, 10, CancellationToken.None)) { }
        });
        Assert.Contains("does not support audio input", pipelineError.Message);
    }

    [Fact]
    public async Task LoadModel_NamingAFileThatIsNotHosted_Is400WithTheOkFalseShape()
    {
        Fixture f = Build();

        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => f.Service.LoadModelAsync(Json("""{"model":"other.gguf"}"""), CancellationToken.None));

        Assert.Equal(400, ex.StatusCode);
        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(ex.Payload));
        Assert.False(doc.RootElement.GetProperty("ok").GetBoolean());
        Assert.False(string.IsNullOrEmpty(doc.RootElement.GetProperty("error").GetString()));
    }

    // ---- upload ----------------------------------------------------------------

    [Fact]
    public async Task Upload_SmallTextFile_ReturnsTheTextContractAndKeepsTheFile()
    {
        Fixture f = Build();
        byte[] bytes = Encoding.UTF8.GetBytes("hello, world");
        using var stream = new MemoryStream(bytes);

        object reply = await f.Service.UploadAsync(stream, "notes.txt", bytes.Length, CancellationToken.None);

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(reply));
        JsonElement root = doc.RootElement;
        Assert.True(root.GetProperty("ok").GetBoolean());
        Assert.Equal("text", root.GetProperty("mediaType").GetString());
        Assert.Equal("notes.txt", root.GetProperty("fileName").GetString());
        Assert.Equal("hello, world", root.GetProperty("textContent").GetString());
        Assert.False(root.GetProperty("truncated").GetBoolean());
        Assert.Equal(JsonValueKind.Null, root.GetProperty("modelContextLimit").ValueKind);
        string file = root.GetProperty("file").GetString();
        Assert.EndsWith(".txt", file);
        Assert.Equal("/uploads/" + file, root.GetProperty("url").GetString());
        Assert.Equal("hello, world", File.ReadAllText(Path.Combine(_baseDir, file)));
        Assert.Equal(bytes.Length, f.Uploads.UsedBytes);
        // The property ORDER is part of the contract the page reads; pin it.
        Assert.Equal(
            new[]
            {
                "ok", "file", "url", "mediaType", "fileName", "textContent", "truncated",
                "truncateLimit", "truncateUnit", "modelContextLimit", "originalTokenCount", "returnedTokenCount",
            },
            root.EnumerateObject().Select(p => p.Name).ToArray());
    }

    [Fact]
    public async Task Upload_Csv_ReturnsAFileBackedContractWithoutCopyingRowsIntoTheReply()
    {
        Fixture f = Build();
        const int reportedUploadBytes = 105 * 1024;
        const string head = "name,region,score\nalice,west,17\n";
        const string tail = "\nbob,east,23000\n";
        byte[] bytes = Encoding.UTF8.GetBytes(
            head + new string('7', reportedUploadBytes - head.Length - tail.Length) + tail);
        Assert.Equal(reportedUploadBytes, bytes.Length);
        using var stream = new MemoryStream(bytes);

        object reply = await f.Service.UploadAsync(stream, "responses.csv", bytes.Length, CancellationToken.None);

        using var doc = JsonDocument.Parse(JsonSerializer.Serialize(reply));
        JsonElement root = doc.RootElement;
        Assert.True(root.GetProperty("ok").GetBoolean());
        Assert.Equal("text", root.GetProperty("mediaType").GetString());
        Assert.Equal("responses.csv", root.GetProperty("fileName").GetString());
        Assert.True(root.GetProperty("fileBacked").GetBoolean());
        Assert.False(root.TryGetProperty("textContent", out _));
        Assert.False(root.GetProperty("truncated").GetBoolean());

        string file = root.GetProperty("file").GetString()!;
        Assert.EndsWith(".csv", file, StringComparison.Ordinal);
        Assert.Equal("/uploads/" + file, root.GetProperty("url").GetString());
        Assert.Equal(bytes, File.ReadAllBytes(Path.Combine(_baseDir, file)));
        Assert.Equal(bytes.Length, f.Uploads.UsedBytes);
        Assert.Equal(
            new[]
            {
                "ok", "file", "url", "mediaType", "fileName", "fileBacked", "truncated",
                "truncateLimit", "truncateUnit", "modelContextLimit", "originalTokenCount", "returnedTokenCount",
            },
            root.EnumerateObject().Select(p => p.Name).ToArray());
    }

    [Fact]
    public async Task Upload_UnsupportedExtension_Is400WritesNothingAndReturnsTheReservation()
    {
        Fixture f = Build();
        using var stream = new MemoryStream(new byte[] { 1, 2, 3 });

        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => f.Service.UploadAsync(stream, "tool.exe", 3, CancellationToken.None));

        Assert.Equal(400, ex.StatusCode);
        Assert.Contains("'.exe'", ex.Message);
        Assert.Equal(0, f.Uploads.UsedBytes);
        Assert.Empty(Directory.GetFiles(_baseDir));
    }

    [Fact]
    public async Task Upload_OverThePerFileCap_Is413()
    {
        Fixture f = Build(new UploadStoragePolicy(_baseDir, maxFileBytes: 4));
        using var stream = new MemoryStream(new byte[10]);

        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(
            () => f.Service.UploadAsync(stream, "big.txt", 10, CancellationToken.None));

        Assert.Equal(413, ex.StatusCode);
        Assert.Empty(Directory.GetFiles(_baseDir));
    }

    // ---- refusals shared with the image / video routes --------------------------

    [Fact]
    public void ImageEditAndVideo_WithoutTheRightModel_AreRefusedWithTheAdapterMessages()
    {
        Fixture f = Build();

        var edit = Assert.Throws<WebUiRequestRejectedException>(() => f.Service.EnsureImageEditAvailable());
        Assert.Equal(400, edit.StatusCode);
        Assert.Equal("""{"error":"The loaded model is not a Qwen-Image-Edit model."}""", JsonSerializer.Serialize(edit.Payload));

        var video = Assert.Throws<WebUiRequestRejectedException>(() => f.Service.EnsureVideoGenerationAvailable());
        Assert.Equal(400, video.StatusCode);
        Assert.Equal("""{"error":"The loaded model is not a video-generation model."}""", JsonSerializer.Serialize(video.Payload));
    }

    [Fact]
    public async Task ImageEditStream_WithoutTheModel_IsOneDoneErrorFrame()
    {
        Fixture f = Build();
        var frames = new List<object>();

        await foreach (object frame in f.Service.ImageEditStreamAsync(Json("""{"prompt":"x"}"""), CancellationToken.None))
            frames.Add(frame);

        Assert.Single(frames);
        Assert.Equal("""{"done":true,"error":"The loaded model is not a Qwen-Image-Edit model."}""", JsonSerializer.Serialize(frames[0]));
    }
}
