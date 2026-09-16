// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

/// <summary>
/// The three stateless chat adapters must give all internally serviced tool rounds one
/// request-private workspace and release it only after the response path has returned.
/// These tests stop at the hosted-model guard, after planning but before inference, so
/// they exercise the real adapter lifetime without loading model weights.
/// </summary>
public sealed class ProtocolAdapterRequestWorkspaceTests : IDisposable
{
    private readonly string _base = Path.Combine(
        Path.GetTempPath(), "ts-adapter-workspace-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
    }

    [Theory]
    [InlineData("openai-chat")]
    [InlineData("openai-responses")]
    [InlineData("ollama-chat")]
    public async Task Request_AcquiresPassesAndReleasesAWorkspace(string protocol)
    {
        string workspaceParent = Path.Combine(_base, protocol, "workspaces");
        string uploadRoot = Path.Combine(_base, protocol, "uploads");
        string hostedModel = Path.Combine(_base, protocol, "hosted.gguf");
        var workspaces = new SessionWorkspaceManager(workspaceParent);
        var runner = new RecordingRunner(workspaceParent);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", hostedModel, "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService();
        using var store = new InMemoryResponsesStore();

        DefaultHttpContext context = ContextFor(RequestBody(protocol));
        await InvokeAdapterAsync(
            protocol, context, service, options, uploadRoot, registry, runner, workspaces, store);

        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
        Assert.Equal(1, runner.DeclareCalls);
        Assert.True(runner.Persists,
            "SkillRequestPlan must receive the lease workspace, which is what makes persists true.");
        Assert.Equal(1, runner.LiveWorkspaceCount);
        Assert.True(runner.WorkspaceWasUsableDuringPlanning);
        string leasedRoot = Assert.IsType<string>(runner.LiveWorkspaceRoot);

        // The adapter's method-scope lease must outlive planning, then disappear on every
        // return path. This request returns a model-not-hosted error immediately after the
        // plan, which also exercises cleanup on failure rather than only the happy path.
        Assert.False(Directory.Exists(leasedRoot));
        Assert.Empty(Directory.EnumerateDirectories(workspaceParent, SessionWorkspace.DirectoryPrefix + "*"));
    }

    private static DefaultHttpContext ContextFor(string body)
    {
        var context = new DefaultHttpContext();
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
        context.Response.Body = new MemoryStream();
        return context;
    }

    [Theory]
    [InlineData("none", 404)]
    [InlineData("auto", 400)]
    [InlineData("required", 400)]
    public async Task JsonFinalAnswer_WithToolHistory_RespectsDisabledToolGeneration(string choice, int expectedStatus)
    {
        var context = ContextFor("""
            {"model":"not-hosted.gguf","messages":[
              {"role":"user","content":"Weather?"},
              {"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Paris\"}"}}]},
              {"role":"tool","tool_call_id":"call_1","content":"sunny"},
              {"role":"user","content":"Return the result as JSON."}],
              "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
              "tool_choice":"CHOICE","response_format":{"type":"json_object"},"stream":false}
            """.Replace("CHOICE", choice));
        var workspaces = new SessionWorkspaceManager(Path.Combine(_base, "workspaces"));
        var runner = new RecordingRunner(Path.Combine(_base, "workspaces"));
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService();
        using var store = new InMemoryResponsesStore();
        await InvokeAdapterAsync("openai-chat", context, service, options, Path.Combine(_base, "uploads"),
            registry, runner, workspaces, store);
        Assert.Equal(expectedStatus, context.Response.StatusCode);
        Assert.Equal(0, runner.DeclareCalls);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        if (choice == "none")
            Assert.DoesNotContain("cannot be combined with tools", response);
        else
            Assert.Contains("cannot be combined with tools", response);
    }

    private static string RequestBody(string protocol) => protocol switch
    {
        "openai-responses" =>
            """{"model":"not-hosted.gguf","input":"fix it","stream":false,"store":false}""",
        "openai-chat" or "ollama-chat" =>
            """{"model":"not-hosted.gguf","messages":[{"role":"user","content":"fix it"}],"stream":false}""",
        _ => throw new ArgumentOutOfRangeException(nameof(protocol)),
    };

    [Theory]
    [InlineData("deepseek41", "json_object", 404, false)]
    [InlineData("deepseek41", "json_schema", 404, false)]
    [InlineData("deepseek41", "json_object", 400, true)]
    [InlineData("deepseek4", "json_object", 400, false)]
    [InlineData("qwen2", "json_schema", 400, false)]
    public async Task ThinkingJson_RequiresProtocolWithDelayedGrammar(string architecture, string format, int status, bool grammarDisabled)
    {
        using var env = new EnvScope();
        env.Set("TS_JSON_GRAMMAR", grammarDisabled ? "0" : null);
        string responseFormat = format == "json_object" ? "{\"type\":\"json_object\"}"
            : "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"result\",\"strict\":true,\"schema\":{\"type\":\"object\",\"properties\":{\"answer\":{\"type\":\"integer\"}},\"required\":[\"answer\"],\"additionalProperties\":false}}}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Return the answer as JSON.\"}],\"think\":true,\"response_format\":" + responseFormat + "}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = architecture };
        using var store = new InMemoryResponsesStore();
        await InvokeAdapterAsync("openai-chat", context, service, options, Path.Combine(_base, "uploads"), registry,
            new RecordingRunner(Path.Combine(_base, "workspaces")), new SessionWorkspaceManager(Path.Combine(_base, "workspaces")), store);
        Assert.Equal(status, context.Response.StatusCode);
    }

    [Theory]
    [InlineData("input_audio", false)]
    [InlineData("input_audio", true)]
    [InlineData("audio_url", false)]
    [InlineData("audio_url", true)]
    public async Task DeepSeek41Audio_Returns400BeforeUploadingAnyMixedMedia(string audioType, bool withImage)
    {
        string image = withImage ? "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AQ==\"}}," : "";
        string audio = audioType == "input_audio"
            ? "{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}"
            : "{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,AQID\"}}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":[" + image + audio + "]}]}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = "deepseek41" };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        await InvokeAdapterAsync("openai-chat", context, service, options, uploads, registry,
            new RecordingRunner(Path.Combine(_base, "workspaces")), new SessionWorkspaceManager(Path.Combine(_base, "workspaces")), store);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        Assert.Contains("does not support audio input", response);
        Assert.Empty(Directory.EnumerateFiles(uploads));
    }

    /// <summary>
    /// Nemotron-H has no audio tower (the Omni mmproj is vision-only), so the
    /// same request-level gate answers 400 with the family's own reason before
    /// any upload is written - for every registered alias of the family.
    /// </summary>
    [Theory]
    [InlineData("nemotron_h_moe", "input_audio", false)]
    [InlineData("nemotron_h_moe", "input_audio", true)]
    [InlineData("nemotron_h_moe", "audio_url", true)]
    [InlineData("nemotron_h", "input_audio", false)]
    [InlineData("nemotron_h_omni", "audio_url", false)]
    public async Task NemotronAudio_Returns400BeforeUploadingAnyMixedMedia(string architecture, string audioType, bool withImage)
    {
        string image = withImage ? "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AQ==\"}}," : "";
        string audio = audioType == "input_audio"
            ? "{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}"
            : "{\"type\":\"audio_url\",\"audio_url\":{\"url\":\"data:audio/wav;base64,AQID\"}}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":[" + image + audio + "]}]}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = architecture };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        await InvokeAdapterAsync("openai-chat", context, service, options, uploads, registry,
            new RecordingRunner(Path.Combine(_base, "workspaces")), new SessionWorkspaceManager(Path.Combine(_base, "workspaces")), store);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        using var error = System.Text.Json.JsonDocument.Parse(response);
        Assert.Equal("invalid_request_error", error.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal(TensorSharp.Models.NemotronModel.AudioInputUnsupportedMessage,
            error.RootElement.GetProperty("error").GetProperty("message").GetString());
        Assert.Empty(Directory.EnumerateFiles(uploads));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public async Task NemotronResponsesAudio_ReturnsJson400BeforeUploadsPlanningOrStreaming(bool malformed, bool stream)
    {
        string audio = malformed
            ? "{\"type\":\"input_audio\"}"
            : "{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"stream\":" + (stream ? "true" : "false") +
            ",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AQID\"}]}," +
            "{\"role\":\"user\",\"content\":[" + audio + "]}]}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = "nemotron_h_moe" };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        string workspaceRoot = Path.Combine(_base, "workspaces");
        var runner = new RecordingRunner(workspaceRoot);

        await InvokeAdapterAsync("openai-responses", context, service, options, uploads, registry,
            runner, new SessionWorkspaceManager(workspaceRoot), store);

        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.StartsWith("application/json", context.Response.ContentType);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        using var error = System.Text.Json.JsonDocument.Parse(response);
        Assert.Equal("invalid_request_error", error.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal(TensorSharp.Models.NemotronModel.AudioInputUnsupportedMessage,
            error.RootElement.GetProperty("error").GetProperty("message").GetString());
        Assert.DoesNotContain("response.created", response);
        Assert.Equal(0, runner.DeclareCalls);
        Assert.Empty(Directory.EnumerateFiles(uploads));
        Assert.False(Directory.Exists(workspaceRoot));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public async Task DeepSeek41ResponsesAudio_ReturnsJson400BeforeUploadsPlanningOrStreaming(bool malformed, bool stream)
    {
        string audio = malformed
            ? "{\"type\":\"input_audio\"}"
            : "{\"type\":\"input_audio\",\"input_audio\":{\"format\":\"wav\",\"data\":\"AQID\"}}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"stream\":" + (stream ? "true" : "false") +
            ",\"input\":[{\"role\":\"user\",\"content\":[{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AQID\"}]}," +
            "{\"role\":\"user\",\"content\":[" + audio + "]}]}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = "deepseek41" };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        string workspaceRoot = Path.Combine(_base, "workspaces");
        var runner = new RecordingRunner(workspaceRoot);

        await InvokeAdapterAsync("openai-responses", context, service, options, uploads, registry,
            runner, new SessionWorkspaceManager(workspaceRoot), store);

        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.StartsWith("application/json", context.Response.ContentType);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        using var error = System.Text.Json.JsonDocument.Parse(response);
        Assert.Equal("invalid_request_error", error.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal(ChatGenerationPipeline.DeepSeek41AudioInputError,
            error.RootElement.GetProperty("error").GetProperty("message").GetString());
        Assert.DoesNotContain("response.created", response);
        Assert.Equal(0, runner.DeclareCalls);
        Assert.Empty(Directory.EnumerateFiles(uploads));
        Assert.False(Directory.Exists(workspaceRoot));
    }

    [Theory]
    [InlineData("openai-chat", "http://example.invalid/photo.png", false)]
    [InlineData("openai-chat", "http://example.invalid/photo.png", true)]
    [InlineData("openai-chat", "https://example.invalid/photo.png", false)]
    [InlineData("openai-chat", "https://example.invalid/photo.png", true)]
    [InlineData("openai-chat", "data:image/png;base64,?", false)]
    [InlineData("openai-chat", "data:image/png;base64,?", true)]
    [InlineData("openai-responses", "http://example.invalid/photo.png", false)]
    [InlineData("openai-responses", "http://example.invalid/photo.png", true)]
    [InlineData("openai-responses", "https://example.invalid/photo.png", false)]
    [InlineData("openai-responses", "https://example.invalid/photo.png", true)]
    [InlineData("openai-responses", "data:image/png;base64,?", false)]
    [InlineData("openai-responses", "data:image/png;base64,?", true)]
    public async Task DeepSeek41InvalidImage_Returns400BeforeUploadsPlanningOrStreaming(string protocol, string url, bool stream)
    {
        object Image(string value) => protocol == "openai-chat"
            ? new { type = "image_url", image_url = new { url = value } }
            : new { type = "input_image", image_url = value };
        string body = System.Text.Json.JsonSerializer.Serialize(new Dictionary<string, object>
        {
            ["model"] = "not-hosted.gguf", ["stream"] = stream,
            [protocol == "openai-chat" ? "messages" : "input"] = new object[]
            {
                new { role = "user", content = new[] { Image("data:image/png;base64,AQID") } },
                new { role = "assistant", content = "An earlier image." },
                new { role = "user", content = new[] { Image(url) } },
            },
        });
        var context = ContextFor(body);
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = "deepseek41" };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        string workspaceRoot = Path.Combine(_base, "workspaces");
        var runner = new RecordingRunner(workspaceRoot);
        await InvokeAdapterAsync(protocol, context, service, options, uploads, registry,
            runner, new SessionWorkspaceManager(workspaceRoot), store);

        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.StartsWith("application/json", context.Response.ContentType);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        using var error = System.Text.Json.JsonDocument.Parse(response);
        Assert.Equal("invalid_request_error", error.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("image_url", error.RootElement.GetProperty("error").GetProperty("message").GetString());
        Assert.DoesNotContain("chat.completion.chunk", response);
        Assert.Equal(0, runner.DeclareCalls);
        Assert.Empty(Directory.EnumerateFiles(uploads));
        Assert.False(Directory.Exists(workspaceRoot));
    }

    [Theory]
    [InlineData("openai-chat", false)]
    [InlineData("openai-chat", true)]
    [InlineData("openai-responses", false)]
    [InlineData("openai-responses", true)]
    public async Task DeepSeek41ValidImageDataUri_ReachesHostedModelGuard(string protocol, bool stream)
    {
        string part = protocol == "openai-chat"
            ? "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AQID\"}}"
            : "{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AQID\"}";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"stream\":" + (stream ? "true" : "false") +
            ",\"" + (protocol == "openai-chat" ? "messages" : "input") + "\":[{\"role\":\"user\",\"content\":[" + part + "]}]}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = "deepseek41" };
        using var store = new InMemoryResponsesStore();
        string uploads = Path.Combine(_base, "uploads");
        await InvokeAdapterAsync(protocol, context, service, options, uploads, registry,
            new RecordingRunner(Path.Combine(_base, "workspaces")), new SessionWorkspaceManager(Path.Combine(_base, "workspaces")), store);
        Assert.Equal(stream ? StatusCodes.Status200OK : StatusCodes.Status404NotFound, context.Response.StatusCode);
        Assert.StartsWith(stream ? "text/event-stream" : "application/json", context.Response.ContentType);
        context.Response.Body.Position = 0;
        string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
        Assert.DoesNotContain("image_url", response);
        Assert.Equal(new byte[] { 1, 2, 3 }, File.ReadAllBytes(Assert.Single(Directory.EnumerateFiles(uploads))));
    }

    [Theory]
    [InlineData("deepseek41", "unknown-choice", false, 400)]
    [InlineData("deepseek41", "unknown-choice", true, 400)]
    [InlineData("deepseek41", "unknown-function", false, 400)]
    [InlineData("deepseek41", "unsupported-schema", true, 400)]
    [InlineData("deepseek41", "bad-parallel", false, 400)]
    [InlineData("deepseek41", "required-without-tools", false, 400)]
    [InlineData("deepseek41", "required", false, 404)]
    [InlineData("deepseek41", "named", false, 404)]
    [InlineData("deepseek41", "none-unused-schema", false, 404)]
    [InlineData("deepseek4", "unknown-choice", false, 404)]
    [InlineData("qwen2", "unsupported-schema", false, 404)]
    public async Task DeepSeek41ToolGrammar_ValidatesPoliciesBeforeStreamingOrModelGeneration(
        string architecture, string scenario, bool stream, int expectedStatus)
    {
        string choice = scenario switch
        {
            "unknown-choice" => "\"bogus\"",
            "unknown-function" => "{\"type\":\"function\",\"function\":{\"name\":\"undeclared\"}}",
            "named" => "{\"type\":\"function\",\"function\":{\"name\":\"weather\"}}",
            "none-unused-schema" => "\"none\"",
            "required" or "required-without-tools" => "\"required\"",
            _ => "\"auto\"",
        };
        string schema = scenario is "unsupported-schema" or "none-unused-schema"
            ? "{\"type\":\"string\",\"pattern\":\"x.*\"}" : "{\"type\":\"string\"}";
        string tools = scenario == "required-without-tools" ? "[]"
            : "[{\"type\":\"function\",\"function\":{\"name\":\"weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":" + schema + "},\"required\":[\"city\"]}}}]";
        string parallel = scenario == "bad-parallel" ? "\"false\"" : "false";
        var context = ContextFor("{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"tools\":" + tools +
            ",\"tool_choice\":" + choice + ",\"parallel_tool_calls\":" + parallel + ",\"stream\":" + (stream ? "true" : "false") + "}");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_base, "hosted.gguf"), "--no-skills" }, _base);
        using var service = new ToolCapableUnloadedModelService { SelectedArchitecture = architecture };
        using var store = new InMemoryResponsesStore();
        await InvokeAdapterAsync("openai-chat", context, service, options, Path.Combine(_base, "uploads"), registry,
            new RecordingRunner(Path.Combine(_base, "workspaces")), new SessionWorkspaceManager(Path.Combine(_base, "workspaces")), store);
        Assert.Equal(expectedStatus, context.Response.StatusCode);
        if (expectedStatus == 400)
        {
            Assert.StartsWith("application/json", context.Response.ContentType);
            context.Response.Body.Position = 0;
            string response = await new StreamReader(context.Response.Body).ReadToEndAsync();
            Assert.Contains("invalid_request_error", response);
            Assert.DoesNotContain("data:", response);
        }
    }

    private static async Task InvokeAdapterAsync(
        string protocol,
        HttpContext context,
        ModelService service,
        ServerHostingOptions options,
        string uploadRoot,
        SkillRegistry registry,
        ICodeRunner runner,
        SessionWorkspaceManager workspaces,
        IResponsesStore store)
    {
        var queue = new InferenceQueue();
        var uploads = new UploadStoragePolicy(uploadRoot);

        switch (protocol)
        {
            case "openai-chat":
                await new OpenAIChatAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance).ChatCompletionsAsync(context);
                break;
            case "openai-responses":
                await new OpenAIResponsesAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance, store).CreateResponseAsync(context);
                break;
            case "ollama-chat":
                await new OllamaAdapter(
                    service, queue, options, uploads, registry, runner, workspaces,
                    NullLoggerFactory.Instance).ChatAsync(context);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(protocol));
        }
    }

    /// <summary>
    /// Only the model-family capability is relevant before HostedModelGuard rejects this
    /// request. No generation API is replaced or reached.
    /// </summary>
    private sealed class ToolCapableUnloadedModelService : ModelService
    {
        public string SelectedArchitecture { get; set; } = "qwen2";
        public override string Architecture => SelectedArchitecture;
    }

    private sealed class RecordingRunner : ICodeRunner
    {
        private readonly string _workspaceParent;

        public RecordingRunner(string workspaceParent) => _workspaceParent = workspaceParent;

        public int DeclareCalls { get; private set; }
        public bool Persists { get; private set; }
        public int LiveWorkspaceCount { get; private set; }
        public string? LiveWorkspaceRoot { get; private set; }
        public bool WorkspaceWasUsableDuringPlanning { get; private set; }

        public bool CanRun => true;
        public string? UnavailableReason => null;

        public ToolFunction Declare() =>
            new() { Name = SkillToolNames.Shell, Description = "runs commands" };

        public IReadOnlyList<ToolFunction> DeclareTools(bool persists)
        {
            DeclareCalls++;
            Persists = persists;
            string[] roots = Directory.Exists(_workspaceParent)
                ? Directory.GetDirectories(
                    _workspaceParent, SessionWorkspace.DirectoryPrefix + "*", SearchOption.TopDirectoryOnly)
                : Array.Empty<string>();
            LiveWorkspaceCount = roots.Length;
            LiveWorkspaceRoot = roots.Length == 1 ? roots[0] : null;
            WorkspaceWasUsableDuringPlanning = LiveWorkspaceRoot != null
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "work"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "env"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "state"))
                && Directory.Exists(Path.Combine(LiveWorkspaceRoot, "tmp"));
            return new[] { Declare() };
        }

        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile>? inputFiles = null,
            Action<string>? onOutput = null,
            SessionWorkspace? workspace = null,
            IReadOnlyList<string>? skillDirectories = null) =>
            SkillToolResult.Failure("inference is not reached by this test");
    }
}
