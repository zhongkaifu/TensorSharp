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
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

/// <summary>
/// Request policies the OpenAI chat adapter settles BEFORE admission, exercised on
/// the real adapter without a model: tools on a diffusion model, the
/// <c>reasoning_effort</c> field, and <c>response_format</c> with <c>think=true</c> on
/// GPT-OSS. A request that passes reaches the hosted-model guard (404 blocking, an SSE
/// error chunk streaming); a rejected one never enters the queue.
/// </summary>
public sealed class OpenAIRequestPolicyTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-request-policy-" + Guid.NewGuid().ToString("N"));
    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private const string ProbeTools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"probe\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";

    [Theory]
    [InlineData(false, ProbeTools + ",\"tool_choice\":\"required\"")]
    [InlineData(true, ProbeTools + ",\"tool_choice\":\"required\"")]
    [InlineData(false, ProbeTools)]
    [InlineData(false, "\"tool_choice\":\"auto\"")]
    [InlineData(true, "\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"probe\"}}")]
    public async Task DiffusionModel_RefusesToolsAndToolChoice_With400(bool stream, string extra)
    {
        var (context, queue) = await Invoke(new DiffusionService(), stream, extra);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.Equal(0, queue.TotalProcessed);
        using var parsed = JsonDocument.Parse(await Response(context));
        Assert.Equal("invalid_request_error", parsed.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("block diffusion", parsed.RootElement.GetProperty("error").GetProperty("message").GetString());
    }

    [Theory]
    [InlineData(false, ProbeTools + ",\"tool_choice\":\"none\"")]
    [InlineData(false, "\"tool_choice\":null")]
    [InlineData(false, "\"max_tokens\":8")]
    public async Task DiffusionModel_StillAnswersPlainRequests(bool stream, string extra)
    {
        var (context, queue) = await Invoke(new DiffusionService(), stream, extra);
        Assert.Equal(1, queue.TotalProcessed);
        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
        Assert.Contains("not hosted by this server", await Response(context));
    }

    [Fact]
    public async Task AutoregressiveModel_KeepsItsTools()
    {
        var (context, queue) = await Invoke(new UnloadedService("gemma4"), false, ProbeTools + ",\"tool_choice\":\"required\"");
        Assert.Equal(1, queue.TotalProcessed);
        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
    }

    [Theory]
    [InlineData("\"reasoning_effort\":\"max\"")]
    [InlineData("\"reasoning_effort\":2")]
    public async Task UnknownReasoningEffort_Is400(string extra)
    {
        var (context, queue) = await Invoke(new UnloadedService("gpt-oss"), false, extra);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.Equal(0, queue.TotalProcessed);
        Assert.Contains("reasoning_effort", await Response(context));
    }

    [Theory]
    [InlineData("\"reasoning_effort\":\"low\"")]
    [InlineData("\"reasoning_effort\":\"high\",\"think\":false")]
    [InlineData("\"think\":false")]
    public async Task KnownReasoningEffort_Passes(string extra)
    {
        var (context, queue) = await Invoke(new UnloadedService("gpt-oss"), false, extra);
        Assert.Equal(1, queue.TotalProcessed);
        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
    }

    private const string JsonObject = "\"response_format\":{\"type\":\"json_object\"},\"think\":true";

    [Theory]
    [InlineData("gpt-oss")]
    [InlineData("gptoss")]
    public async Task Harmony_AcceptsResponseFormatWithThinking(string architecture)
    {
        var (context, queue) = await Invoke(new UnloadedService(architecture), false, JsonObject);
        Assert.Equal(1, queue.TotalProcessed);
        Assert.Equal(StatusCodes.Status404NotFound, context.Response.StatusCode);
    }

    [Fact]
    public async Task FamilyWithoutADelayedThinkingTrigger_StillRefusesResponseFormatWithThinking()
    {
        var (context, queue) = await Invoke(new UnloadedService("unknown-test-architecture"), false, JsonObject);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        Assert.Equal(0, queue.TotalProcessed);
        Assert.Contains("cannot be combined with think=true", await Response(context));
    }

    private async Task<(DefaultHttpContext, InferenceQueue)> Invoke(ModelService service, bool stream, string extra)
    {
        var context = new DefaultHttpContext();
        string request = "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"Call probe now.\"}],\"stream\":"
            + (stream ? "true" : "false") + (extra.Length == 0 ? "" : "," + extra) + "}";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var queue = new InferenceQueue();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        using (service)
        {
            await new OpenAIChatAdapter(service, queue, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry,
                null, new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        }
        return (context, queue);
    }

    private static async Task<string> Response(DefaultHttpContext context)
    { context.Response.Body.Position = 0; return await new StreamReader(context.Response.Body).ReadToEndAsync(); }

    private sealed class UnloadedService(string architecture) : ModelService { public override string Architecture => architecture; }
    private sealed class DiffusionService : ModelService
    {
        public override string Architecture => "diffusion-gemma";
        public override bool IsDiffusionModel => true;
    }
}
