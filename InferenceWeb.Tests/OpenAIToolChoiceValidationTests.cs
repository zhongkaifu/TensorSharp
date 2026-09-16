// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

/// <summary>Exercise the actual adapter before admission, without a model/backend.</summary>
public sealed class OpenAIToolChoiceValidationTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-tool-choice-" + Guid.NewGuid().ToString("N"));
    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    public static IEnumerable<object[]> InvalidChoices()
    {
        foreach (string architecture in new[] { "qwen4exp", "deepseek41", "unknown-test-architecture" })
        foreach (bool stream in new[] { false, true })
        foreach (string scenario in new[] { "required-absent", "required-empty", "named-undeclared" })
            yield return new object[] { architecture, stream, scenario };
    }

    [Theory]
    [MemberData(nameof(InvalidChoices))]
    public async Task ImpossibleClientToolChoice_Returns400BeforeQueueOrStreaming(string architecture, bool stream, string scenario)
    {
        string choice = scenario == "named-undeclared"
            ? "{\"type\":\"function\",\"function\":{\"name\":\"undeclared_weather_tool\"}}" : "\"required\"";
        string tools = scenario switch { "required-absent" => "", "required-empty" => ",\"tools\":[]", _ => WeatherTools };
        var (context, queue, service, runner) = await Invoke(architecture, stream, choice, tools);
        using (service)
        {
            Assert.Null(service.Model);
            Assert.Equal(0, queue.TotalProcessed);
            Assert.Equal(0, runner.ExecuteCalls);
            Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
            Assert.StartsWith("application/json", context.Response.ContentType);
            string response = await Response(context);
            using var parsed = JsonDocument.Parse(response);
            Assert.Equal("invalid_request_error", parsed.RootElement.GetProperty("error").GetProperty("type").GetString());
            Assert.Contains(scenario == "named-undeclared" ? "undeclared_weather_tool" : "at least one",
                parsed.RootElement.GetProperty("error").GetProperty("message").GetString());
            Assert.DoesNotContain("data:", response);
        }
    }

    [Theory]
    [InlineData(false, "required")]
    [InlineData(true, "required")]
    [InlineData(false, "named")]
    [InlineData(true, "named")]
    [InlineData(false, "none")]
    [InlineData(true, "none")]
    public async Task ValidClientChoice_ReachesExistingHostedModelGuard(bool stream, string scenario)
    {
        string choice = scenario == "named" ? "{\"type\":\"function\",\"function\":{\"name\":\"get_weather\"}}" : JsonSerializer.Serialize(scenario);
        var (context, queue, service, runner) = await Invoke("qwen4exp", stream, choice, WeatherTools);
        using (service)
        {
            Assert.Null(service.Model);
            Assert.Equal(1, queue.TotalProcessed);
            Assert.Equal(0, runner.ExecuteCalls);
            Assert.Equal(stream ? StatusCodes.Status200OK : StatusCodes.Status404NotFound, context.Response.StatusCode);
            Assert.Contains("not hosted by this server", await Response(context));
        }
    }

    [Theory]
    [InlineData(false, null)]
    [InlineData(true, null)]
    [InlineData(false, "auto")]
    [InlineData(true, "auto")]
    public async Task InternalTools_RemainAvailableWithoutClientDeclarations(bool stream, string? choice)
    {
        var (context, queue, service, runner) = await Invoke("qwen2", stream, choice == null ? null : JsonSerializer.Serialize(choice), "", codeEnabled: true);
        using (service)
        {
            Assert.Null(service.Model);
            Assert.Equal(1, queue.TotalProcessed);
            Assert.Equal(1, runner.DeclareCalls);
            Assert.Equal(0, runner.ExecuteCalls);
            Assert.Equal(stream ? StatusCodes.Status200OK : StatusCodes.Status404NotFound, context.Response.StatusCode);
            Assert.Contains("not hosted by this server", await Response(context));
        }
    }

    [Theory]
    [InlineData(false, "\"required\"")]
    [InlineData(true, "\"required\"")]
    [InlineData(false, "{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}")]
    [InlineData(true, "{\"type\":\"function\",\"function\":{\"name\":\"shell\"}}")]
    public async Task InternalToolCannotSilentlySatisfyExplicitClientContract(bool stream, string choice)
    {
        var (context, queue, service, runner) = await Invoke("qwen2", stream, choice, "", codeEnabled: true);
        using (service)
        {
            Assert.Null(service.Model);
            Assert.Equal(1, runner.DeclareCalls); // a real effective internal tool exists
            Assert.Equal(0, queue.TotalProcessed);
            Assert.Equal(0, runner.ExecuteCalls);
            Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
            Assert.Contains("at least one client-declared function", await Response(context));
        }
    }

    private const string WeatherTools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}]";
    private async Task<(DefaultHttpContext, InferenceQueue, UnloadedService, RecordingRunner)> Invoke(
        string architecture, bool stream, string? choice, string tools, bool codeEnabled = false)
    {
        var context = new DefaultHttpContext();
        string request = "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Weather?\"}],\"stream\":"
            + (stream ? "true" : "false") + tools + (choice == null ? "" : ",\"tool_choice\":" + choice) + "}";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var queue = new InferenceQueue();
        var service = new UnloadedService(architecture);
        var runner = new RecordingRunner();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        await new OpenAIChatAdapter(service, queue, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry,
            codeEnabled ? runner : null, new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        return (context, queue, service, runner);
    }
    private static async Task<string> Response(DefaultHttpContext context)
    { context.Response.Body.Position = 0; return await new StreamReader(context.Response.Body).ReadToEndAsync(); }
    private sealed class UnloadedService(string architecture) : ModelService { public override string Architecture => architecture; }
    private sealed class RecordingRunner : ICodeRunner
    {
        public int DeclareCalls { get; private set; }
        public int ExecuteCalls { get; private set; }
        public bool CanRun => true;
        public string? UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell", Description = "Internal shell tool" };
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists) { DeclareCalls++; return new[] { Declare() }; }
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile>? inputFiles = null, Action<string>? onOutput = null,
            SessionWorkspace? workspace = null, IReadOnlyList<string>? skillDirectories = null)
        { ExecuteCalls++; throw new InvalidOperationException("No inference/tool execution is allowed in this fixture"); }
    }
}
