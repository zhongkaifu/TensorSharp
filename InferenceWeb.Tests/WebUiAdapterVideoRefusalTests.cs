// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using static InferenceWeb.Tests.WebUiVideoLoraTests;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>POST /v1/videos/generations</c> answers a request the loaded video model refuses with a
/// 400 in the route's own envelope (<c>{ error: { message, type } }</c>), the model's text
/// intact; it used to let the exception escape as a 500, which is what a MiniMax-H3 Turbo
/// plug-in's step-count refusal (and the base model's guidance refusal) became. The service's
/// own refusals keep their status, and cancellation still propagates. A failure on the server's
/// side (an <see cref="InvalidOperationException"/>: a forward that failed, a model unloaded
/// under the request) stays a 5xx, now a <c>server_error</c> with its text, never a 400 an
/// OpenAI client would not retry. A stand-in model with the plug-in contract and no weights;
/// see <see cref="WebUiVideoLoraTests"/>.
/// </summary>
public sealed class WebUiAdapterVideoRefusalTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-video-route-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();
    private readonly MiniMaxH3LoraFixtures _fixtures = new();
    private readonly ModelService _models;
    private readonly TurboStandIn _model;
    private readonly WebUiAdapter _adapter;

    public WebUiAdapterVideoRefusalTests()
    {
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
        _env.Set(LoraCliFlags.EnvironmentVariable, null);
        _fixtures.WriteText(MiniMaxH3LoraFixtures.Weights8, "stand-in weights, never read");
        _model = new TurboStandIn(WriteStandInGguf(_dir, "turbo.gguf"));
        _models = new ModelService(NullLogger<ModelService>.Instance, (_, _, _, _) => _model);
        _models.LoadModel(Path.Combine(_dir, "turbo.gguf"), null, "cpu");
        _adapter = new WebUiAdapter(_models, new SessionManager(), Options(_dir),
            new UploadStoragePolicy(Path.Combine(_dir, "uploads")),
            new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }),
            codeRunner: null, workspaces: null, codeArtifacts: null, NullLoggerFactory.Instance);
    }

    public void Dispose()
    {
        _model.ReleaseAll();
        _models.Dispose();
        _fixtures.Dispose();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private void ApplyTurbo8() => _model.SetLoras([new LoraSpec(_fixtures.WriteManifest())]);

    [Theory]
    [InlineData("steps", 20f, StepsRefusal)]
    [InlineData("flowShift", 12f, ShiftRefusal)]
    public async Task ARequestThePluginsRecipeContradictsIsA400WithItsText(string field, float value, string refusal)
    {
        ApplyTurbo8();
        var (status, body) = await PostAsync(new Dictionary<string, object> { ["prompt"] = Prompt, [field] = value });
        Assert.Equal(StatusCodes.Status400BadRequest, status);
        Assert.Equal(refusal, body.GetProperty("error").GetProperty("message").GetString());
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task TheBaseModelsGuidanceRefusalIsA400Too()
    {
        var (status, body) = await PostAsync(new { prompt = Prompt, cfg = 2 });
        Assert.Equal(StatusCodes.Status400BadRequest, status);
        Assert.StartsWith("MiniMax-H3 is CFG-distilled and requires guidance 1.0.", body.GetProperty("error").GetProperty("message").GetString());
        Assert.Equal("invalid_request_error", body.GetProperty("error").GetProperty("type").GetString());
    }

    [Fact]
    public async Task ARequestWithoutAStepCountReachesTheModelWithTheRecipeLeftToIt()
    {
        ApplyTurbo8();
        var (status, body) = await PostAsync(new { prompt = Prompt });
        // The stand-in stops every clip it is given; reaching it is what is under test.
        Assert.Equal(StatusCodes.Status400BadRequest, status);
        Assert.Equal(TurboStandIn.Reached, body.GetProperty("error").GetProperty("message").GetString());
        Assert.Equal(1, _model.GenerateCalls);
        Assert.Equal(0, _model.LastParams!.Steps);
    }

    [Fact]
    public async Task TheServicesOwnRefusalKeepsItsStatusAndBody()
    {
        // Keyframes for a model that takes none: the service refuses before the model runs.
        var (status, body) = await PostAsync(new
        {
            prompt = Prompt,
            image_conditions = new[] { new { image = Convert.ToBase64String(new byte[] { 1, 2, 3 }), frameIndex = 0 } },
        });
        Assert.Equal(StatusCodes.Status400BadRequest, status);
        Assert.Equal("The loaded video model does not accept image keyframes at arbitrary frame positions.",
            body.GetProperty("error").GetString());
        Assert.Equal(0, _model.GenerateCalls);
    }

    [Theory]
    [InlineData("forward")]
    [InlineData("unloaded")]
    public async Task AServerSideFailureIsA500ServerErrorWithItsText(string failure)
    {
        // What a native forward that ran out of device memory throws, and what a model unloaded
        // while the request waited throws (ObjectDisposedException is an InvalidOperationException).
        Exception thrown = failure == "forward"
            ? new InvalidOperationException("MiniMax-H3 DiT forward failed: ggml_metal: out of device memory")
            : new ObjectDisposedException(null, "MiniMax-H3 was unloaded; its LoRA plug-ins were not applied.");
        _model.ThrowOnGenerate = thrown;
        var (status, body) = await PostAsync(new { prompt = Prompt });
        Assert.Equal(StatusCodes.Status500InternalServerError, status);
        Assert.Equal(thrown.Message, body.GetProperty("error").GetProperty("message").GetString());
        Assert.Equal("server_error", body.GetProperty("error").GetProperty("type").GetString());
        Assert.Equal(1, _model.GenerateCalls);
    }

    [Fact]
    public async Task CancellationIsNotARefusal()
    {
        _model.ThrowOnGenerate = new OperationCanceledException();
        await Assert.ThrowsAsync<OperationCanceledException>(() => PostAsync(new { prompt = Prompt }));
    }

    private async Task<(int Status, JsonElement Body)> PostAsync(object request)
    {
        await using var services = new ServiceCollection().AddLogging().BuildServiceProvider();
        var context = new DefaultHttpContext { RequestServices = services };
        context.Request.Method = HttpMethods.Post;
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(JsonSerializer.SerializeToUtf8Bytes(request));
        var response = new MemoryStream();
        context.Response.Body = response;

        IResult result = await _adapter.OpenAIVideoGenerationsAsync(context.Request).WaitAsync(TestTimeout);
        await result.ExecuteAsync(context);

        response.Position = 0;
        using var json = await JsonDocument.ParseAsync(response);
        return (context.Response.StatusCode, json.RootElement.Clone());
    }
}
