using System;
using System.Diagnostics.CodeAnalysis;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using TensorSharp.Models.Embeddings;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Hosting;

/// <summary>Registers a resident embedding model without creating a chat inference engine.</summary>
public static class EmbeddingHosting
{
    public static IServiceCollection AddTensorSharpEmbeddings(this IServiceCollection services, ServerHostingOptions options,
        string? requestedBackend = null)
    {
        if (!options.EmbeddingsEnabled)
            throw new ArgumentException("Embedding hosting requires EmbeddingsEnabled.", nameof(options));
        if (!TryResolveBackend(options, requestedBackend, out string? resolvedBackend, out string? error))
            throw new ArgumentException(error, nameof(requestedBackend));
        string backend = resolvedBackend;
        services.TryAddSingleton(options);
        services.AddSingleton<IEmbeddingModel>(_ => EmbeddingModel.Load(options.StartupModelPath,
            new EmbeddingModelOptions { Backend = backend, Threads = options.EmbeddingThreads,
                MaxTokens = options.EmbeddingContextSize }));
        services.AddSingleton<EmbeddingAdapter>();
        return services;
    }

    /// <summary>
    /// The encoder backend an embedding deployment loads on, or false with the
    /// operator-facing reason. An explicitly requested backend this machine does not have
    /// is refused rather than replaced; a null request takes the resolved default. Lets
    /// the host report that refusal as one line before it registers any service.
    /// </summary>
    public static bool TryResolveBackend(ServerHostingOptions options, string? requestedBackend,
        [NotNullWhen(true)] out string? modelBackend, [NotNullWhen(false)] out string? error)
    {
        ArgumentNullException.ThrowIfNull(options);
        modelBackend = null;
        string selectedBackend = options.DefaultBackend;
        if (!string.IsNullOrWhiteSpace(requestedBackend)
            && !BackendSelector.TryResolveSupportedBackend(options, requestedBackend, out selectedBackend, out string selectError))
        {
            error = selectError;
            return false;
        }
        try
        {
            modelBackend = ResolveModelBackend(selectedBackend);
        }
        catch (ArgumentException ex)
        {
            error = ex.Message;
            return false;
        }
        error = null;
        return true;
    }

    internal static string ResolveModelBackend(string backend) => BackendCatalog.Canonicalize(backend) switch
    {
        "cpu" => "CPU",
        "ggml_cpu" => "GGML_CPU",
        "ggml_metal" => "GGML_METAL",
        "ggml_cuda" => "GGML_CUDA",
        _ => throw new ArgumentException($"Embedding hosting requires cpu (pure C#), ggml_cpu, ggml_metal, or ggml_cuda; got '{backend}'."),
    };

    /// <summary>Rejects generation requests before any adapter attempts to load an encoder as a chat model.</summary>
    public static IApplicationBuilder UseEmbeddingModelGuard(this IApplicationBuilder app) => app.Use(async (context, next) =>
    {
        if (HttpMethods.IsPost(context.Request.Method) && IsGenerationPath(context.Request.Path)
            && context.RequestServices.GetService<ServerHostingOptions>()?.EmbeddingsEnabled == true)
        {
            await EmbeddingAdapter.WriteErrorAsync(context,
                "This server hosts an embedding model. Use /v1/embeddings or /api/embed.").ConfigureAwait(false);
            return;
        }
        await next(context).ConfigureAwait(false);
    });

    /// <summary>
    /// Every mapped POST route that runs (or loads) a generation model. Kept to routes that
    /// exist: the image, video and Jev routes were missing, so in embedding mode they fell
    /// through to adapters that answered "the loaded model is not a Qwen-Image-2.1 model"
    /// about a server that had loaded an encoder, while a /v1/completions entry guarded a
    /// route the server never maps.
    /// </summary>
    internal static bool IsGenerationPath(PathString path) => path.Value?.TrimEnd('/').ToLowerInvariant() is
        "/v1/chat/completions" or "/v1/responses" or "/v1/systemone" or "/v1/videos/generations"
        or "/api/generate" or "/api/chat" or "/api/chat/ollama" or "/api/models/load"
        or "/api/image-generate" or "/api/image-generate/stream"
        or "/api/image-edit" or "/api/image-edit/stream"
        or "/api/video-generate" or "/api/video-generate/stream";

    internal static Task InvokeAsync(HttpContext context, Func<EmbeddingAdapter, HttpContext, Task> action)
    {
        var adapter = context.RequestServices.GetService<EmbeddingAdapter>();
        return adapter != null ? action(adapter, context) : EmbeddingAdapter.WriteErrorAsync(context,
            "This server does not host an embedding model. Start it with --model <encoder.gguf> --embeddings.");
    }
}
