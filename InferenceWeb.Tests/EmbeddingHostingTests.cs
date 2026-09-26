using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using System.Text.Json;
using TensorSharp.Models.Embeddings;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

public class EmbeddingHostingTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-embedding-options-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }

    [Fact]
    public void EmbeddingOptions_AreAcceptedByEveryArgumentPass()
    {
        string[] args = ["--model", "encoder.gguf", "--embeddings", "--embedding-threads=4",
            "--embedding-context-size", "4096", "--backend", "ggml_cpu", "--no-skills"];
        ServerHostingOptions options = ServerOptionsBuilder.Build(args, _directory);
        Assert.True(options.EmbeddingsEnabled);
        Assert.Equal(4, options.EmbeddingThreads);
        Assert.Equal(4096, options.EmbeddingContextSize);
        Assert.EndsWith("encoder.gguf", options.StartupModelPath);
        Assert.Equal("ggml_cpu", ServerOptionsBuilder.ReadConfiguredBackendInput(args));
        Assert.False(ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(args));
    }

    [Fact]
    public void JsonConfig_EnablesEmbeddings_AndCommandLineOverridesLimits()
    {
        Directory.CreateDirectory(_directory);
        string config = Path.Combine(_directory, "embedding-server.json");
        File.WriteAllText(config, """
            { "model": "encoder.gguf", "embeddings": true, "embedding-threads": 4,
              "embedding-context-size": 4096, "no-skills": true }
            """);
        string[] args = ConfigFileArgs.Expand(["--config", config, "--embedding-threads", "8"]);
        ServerHostingOptions options = ServerOptionsBuilder.Build(args, _directory);
        Assert.True(options.EmbeddingsEnabled);
        Assert.Equal(8, options.EmbeddingThreads);
        Assert.Equal(4096, options.EmbeddingContextSize);
    }

    [Theory]
    [InlineData("--embeddings")]
    [InlineData("--embedding-threads", "4")]
    [InlineData("--embedding-context-size", "4096")]
    [InlineData("--embeddings", "--model", "model.gguf", "--mmproj", "projector.gguf")]
    [InlineData("--embeddings", "--model", "model.gguf", "--embedding-threads", "-1")]
    [InlineData("--embeddings", "--model", "model.gguf", "--embedding-context-size", "zero")]
    public void InvalidEmbeddingOptions_FailAtStartup(params string[] args)
    {
        Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(args, _directory));
    }

    [Fact]
    public void DefaultHosting_RemainsGenerationMode()
    {
        ServerHostingOptions options = ServerOptionsBuilder.Build(["--no-skills"], _directory);
        Assert.False(options.EmbeddingsEnabled);
        Assert.Equal(0, options.EmbeddingThreads);
        Assert.Equal(0, options.EmbeddingContextSize);
        Assert.Throws<ArgumentException>(() => new ServiceCollection().AddTensorSharpEmbeddings(options));
    }

    [Fact]
    public void Help_DescribesEmbeddingModeAndResourceOptions()
    {
        using var text = new StringWriter();
        ServerUsage.PrintUsage(text);
        Assert.Contains("--embeddings", text.ToString());
        Assert.Contains("--embedding-threads", text.ToString());
        Assert.Contains("--embedding-context-size", text.ToString());
        Assert.Contains("/v1/embeddings", text.ToString());
    }

    [Fact]
    public void ExplicitUnavailableBackend_IsRejectedInsteadOfFallingBack()
    {
        Assert.Throws<ArgumentException>(() => new ServiceCollection()
            .AddTensorSharpEmbeddings(EmbeddingEndpointTests.Options(), "ggml_cuda"));
    }

    [Fact]
    public void ManagedCpuHosting_ResolvesWithoutNativeBackendDiscovery()
    {
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            ["--model", "encoder.gguf", "--embeddings", "--backend", "CPU", "--no-skills"], _directory);
        Assert.Equal("cpu", options.DefaultBackend);
        Assert.True(options.UsesManagedEmbeddingBackend);
        Assert.Equal("cpu", Assert.Single(options.SupportedBackends).Value);
        new ServiceCollection().AddTensorSharpEmbeddings(options, "CPU");
    }

    [Fact]
    public async Task ManagedCpuHosting_ResolvesManagedEncoderAndRunsInference()
    {
        using var fixture = new EmbeddingModelTests.TinyEncoderFixture();
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            ["--model", fixture.Path, "--embeddings", "--backend", "cpu", "--embedding-threads", "1", "--no-skills"], _directory);
        using var services = new ServiceCollection().AddTensorSharpEmbeddings(options, "cpu").BuildServiceProvider();
        var model = Assert.IsType<EmbeddingModel>(services.GetRequiredService<IEmbeddingModel>());
        Assert.True(model.IsManaged);
        Assert.Equal("CPU", model.Backend);
        int[] tokens = model.Tokenize("Hello world");
        EmbeddingBatchResult result = await model.EmbedTokensAsync([tokens]);
        Assert.Equal(tokens.Length, result.PromptTokens);
        float[] embedding = Assert.Single(result.Embeddings);
        Assert.Equal(model.Dimensions, embedding.Length);
        Assert.All(embedding, value => Assert.True(float.IsFinite(value)));
        Assert.InRange(embedding.Sum(value => (double)value * value), 0.99999, 1.00001);
    }

    [Fact]
    public async Task BackendOverride_ReportsActualLoadedBackendWithoutChangingCallerOptions()
    {
        using var fixture = new EmbeddingModelTests.TinyEncoderFixture();
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            ["--model", fixture.Path, "--embeddings", "--backend", "ggml_cpu", "--no-skills"], _directory);
        using var services = new ServiceCollection().AddLogging()
            .AddTensorSharpEmbeddings(options, requestedBackend: "cpu").BuildServiceProvider();
        var model = Assert.IsType<EmbeddingModel>(services.GetRequiredService<IEmbeddingModel>());
        Assert.True(model.IsManaged);
        var context = new DefaultHttpContext { RequestServices = services };
        await using var body = new MemoryStream();
        context.Response.Body = body;
        await services.GetRequiredService<EmbeddingAdapter>().GetWebModels().ExecuteAsync(context);
        body.Position = 0;
        using var json = await JsonDocument.ParseAsync(body);
        Assert.Equal("cpu", json.RootElement.GetProperty("loadedBackend").GetString());
        Assert.Equal("ggml_cpu", json.RootElement.GetProperty("defaultBackend").GetString());
        Assert.Equal("ggml_cpu", options.DefaultBackend);
        Assert.Same(options, services.GetRequiredService<ServerHostingOptions>());
    }

    [Theory]
    [InlineData("cpu", "CPU")]
    [InlineData("ggml_cpu", "GGML_CPU")]
    [InlineData("ggml_metal", "GGML_METAL")]
    [InlineData("ggml_cuda", "GGML_CUDA")]
    public void EmbeddingBackendVocabulary_KeepsManagedAndNativeChoicesDistinct(string hosted, string model)
    {
        Assert.Equal(model, EmbeddingHosting.ResolveModelBackend(hosted));
    }

    [Theory]
    [InlineData("cuda")]
    [InlineData("direct-cuda")]
    [InlineData("mlx")]
    [InlineData("ggml_vulkan")]
    public void UnsupportedEmbeddingBackend_IsRejectedBeforeAvailabilityFallback(string backend)
    {
        var error = Assert.Throws<ArgumentException>(() => ServerOptionsBuilder.Build(
            ["--model", "encoder.gguf", "--embeddings", "--backend", backend, "--no-skills"], _directory));
        Assert.Contains("cpu (pure C#)", error.Message);
    }

    [Theory]
    [InlineData("/v1/chat/completions")]
    [InlineData("/V1/CHAT/COMPLETIONS/")]
    [InlineData("/api/generate/")]
    [InlineData("/api/models/load")]
    public void GenerationGuard_UsesSameCaseAndSlashRulesAsRouting(string path)
    {
        Assert.True(EmbeddingHosting.IsGenerationPath(path));
        Assert.False(EmbeddingHosting.IsGenerationPath("/v1/embeddings"));
    }

    /// <summary>
    /// Every mapped generation POST route, image, video and Jev included. Those three
    /// families were missing, so an embedding server answered them from adapters that
    /// described a model it had never loaded ("the loaded model is not a Qwen-Image-2.1
    /// model").
    /// </summary>
    [Theory]
    [InlineData("/v1/responses")]
    [InlineData("/v1/systemone")]
    [InlineData("/v1/videos/generations")]
    [InlineData("/api/chat")]
    [InlineData("/api/chat/ollama")]
    [InlineData("/api/image-generate")]
    [InlineData("/api/image-generate/stream")]
    [InlineData("/api/image-edit")]
    [InlineData("/api/image-edit/stream")]
    [InlineData("/api/video-generate")]
    [InlineData("/api/video-generate/stream")]
    public void GenerationGuard_CoversEveryGenerationRoute(string path)
    {
        Assert.True(EmbeddingHosting.IsGenerationPath(path));
    }

    [Theory]
    [InlineData("/v1/completions")]   // never mapped: guarding it only implied it exists
    [InlineData("/v1/embeddings")]
    [InlineData("/api/embed")]
    [InlineData("/api/embeddings")]
    [InlineData("/api/upload")]
    [InlineData("/api/show")]
    public void GenerationGuard_LeavesOtherRoutesAlone(string path)
    {
        Assert.False(EmbeddingHosting.IsGenerationPath(path));
    }

    [Theory]
    [InlineData("/api/image-generate", false)]
    [InlineData("/api/video-generate/stream", false)]
    [InlineData("/v1/systemone", true)]
    public async Task GenerationGuard_AnswersImageVideoAndJevWithTheEmbeddingModeError(string path, bool openAIShape)
    {
        using var services = new ServiceCollection().AddSingleton(EmbeddingEndpointTests.Options()).BuildServiceProvider();
        var app = new ApplicationBuilder(services);
        app.UseEmbeddingModelGuard();
        bool reachedEndpoint = false;
        app.Run(_ =>
        {
            reachedEndpoint = true;
            return Task.CompletedTask;
        });
        RequestDelegate pipeline = app.Build();

        var context = new DefaultHttpContext { RequestServices = services };
        context.Request.Method = HttpMethods.Post;
        context.Request.Path = path;
        await using var body = new MemoryStream();
        context.Response.Body = body;

        await pipeline(context);

        Assert.False(reachedEndpoint);
        Assert.Equal(StatusCodes.Status400BadRequest, context.Response.StatusCode);
        body.Position = 0;
        using var json = await JsonDocument.ParseAsync(body);
        string message = openAIShape
            ? json.RootElement.GetProperty("error").GetProperty("message").GetString()
            : json.RootElement.GetProperty("error").GetString();
        Assert.Contains("hosts an embedding model", message);
    }

    [Fact]
    public void EmbeddingStartupBanner_ListsEmbeddingApisAndHonorsHeadlessMode()
    {
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            ["--model", "encoder.gguf", "--embeddings", "--no-webui", "--no-skills"], _directory);
        string endpoints = string.Join('\n', StartupBanner.DescribeEndpoints(options));
        Assert.Contains("/v1/embeddings", endpoints);
        Assert.Contains("/api/embed", endpoints);
        Assert.Contains("/api/embeddings", endpoints);
        Assert.DoesNotContain("Chat", endpoints);
        Assert.DoesNotContain("Generate", endpoints);
        Assert.DoesNotContain("index.html", endpoints);
    }
}
