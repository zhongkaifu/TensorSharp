using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using TensorSharp.Models.Embeddings;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>OpenAI and Ollama embedding protocols over one resident encoder.</summary>
public sealed partial class EmbeddingAdapter : IDisposable, IAsyncDisposable
{
    public const int MaxBatchInputs = 2048;
    public const int MaxBatchTokens = 262144;
    /// <summary>Embedding JSON body limit, enforced before buffering even for chunked requests.</summary>
    public const int MaxRequestBodyBytes = 16 * 1024 * 1024;
    private readonly IEmbeddingModel _model;
    private readonly ServerHostingOptions _options;
    private readonly EmbeddingRequestDispatcher _dispatcher;
    private readonly EmbeddingGemma2MultimodalEncoder? _multimodal;
    /// <summary><c>tensorsharp.backend</c> of the embeddings span and metrics; fixed for the adapter's life.</summary>
    private readonly string? _telemetryBackend;

    public EmbeddingAdapter(IEmbeddingModel model, ServerHostingOptions options, EmbeddingGemma2MultimodalEncoder? multimodal = null)
    {
        _model = model;
        _options = options;
        _dispatcher = new EmbeddingRequestDispatcher(model);
        _multimodal = multimodal;
        _telemetryBackend = ConcreteModel is { } embeddingModel
            ? BackendCatalog.Canonicalize(embeddingModel.Backend) : options.DefaultBackend;
    }

    public void Dispose() => _dispatcher.Dispose();
    public ValueTask DisposeAsync() => _dispatcher.DisposeAsync();

    public Task OpenAIAsync(HttpContext context) => EmbedAsync(context, openAI: true);
    public Task OllamaAsync(HttpContext context) => EmbedAsync(context, openAI: false);

    private EmbeddingModel? ConcreteModel => _multimodal?.TextModel ?? _model as EmbeddingModel;

    public IResult ListModels() => Results.Json(new
    {
        @object = "list",
        data = new[] { new { id = _model.ModelName, @object = "model", created = 0, owned_by = "local",
            capabilities = new[] { "embedding" }, embedding_dimensions = _model.Dimensions,
            context_length = _model.MaxTokens } },
    });

    public IResult GetTags()
    {
        var file = new FileInfo(_options.StartupModelPath);
        return Results.Json(new
        {
            models = new[] { new { name = _model.ModelName, model = _model.ModelName,
                modified_at = file.Exists ? file.LastWriteTimeUtc.ToString("o") : "",
                size = file.Exists ? file.Length : 0, details = new { format = "gguf", family = _model.Architecture },
                capabilities = new[] { "embedding" } } },
        });
    }

    public IResult GetWebModels() => Results.Json(new
    {
        models = new[] { Path.GetFileName(_options.StartupModelPath) },
        mmProjModels = _multimodal == null ? Array.Empty<string>() : new[] { Path.GetFileName(_options.StartupMmProjPath) },
        loaded = Path.GetFileName(_options.StartupModelPath),
        loadedMmProj = _multimodal == null ? null : Path.GetFileName(_options.StartupMmProjPath),
        loadedBackend = ConcreteModel is { } embeddingModel
            ? BackendCatalog.Canonicalize(embeddingModel.Backend) : _options.DefaultBackend,
        defaultBackend = _options.DefaultBackend,
        supportedBackends = _options.SupportedBackends,
        architecture = _model.Architecture,
        contextTokens = _model.MaxTokens,
        modelContextTokens = _model.MaxTokens,
        embeddingDimensions = _model.Dimensions,
        capabilities = new[] { "embedding" },
        visionReady = _multimodal?.SupportsImages == true,
        acceptsVisionProjector = _model.Architecture == "gemma-embedding2",
        defaultMaxTokens = _options.DefaultMaxTokens,
    });

    public async Task ShowAsync(HttpContext context)
    {
        try
        {
            using var document = await ReadBodyAsync(context).ConfigureAwait(false);
            ValidateModel(document.RootElement);
            var metadata = new Dictionary<string, object>
            {
                ["general.architecture"] = _model.Architecture,
                [_model.Architecture + ".embedding_length"] = ConcreteModel?.HiddenSize ?? _model.Dimensions,
                [_model.Architecture + ".context_length"] = _model.MaxTokens,
            };
            if (_model.Architecture == "gemma-embedding2")
                metadata["gemma-embedding2.embedding_length_out"] = _model.Dimensions;
            await context.Response.WriteAsJsonAsync(new { details = new { format = "gguf", family = _model.Architecture },
                model_info = metadata, capabilities = new[] { "embedding" }, parameters = "", template = "" },
                context.RequestAborted).ConfigureAwait(false);
        }
        catch (EmbeddingRequestException ex)
        {
            await WriteErrorAsync(context, ex.Message, ex.Status, ex.Param).ConfigureAwait(false);
        }
        catch (JsonException)
        {
            await WriteErrorAsync(context, "Request body must contain valid JSON.", 400).ConfigureAwait(false);
        }
    }

    private async Task EmbedAsync(HttpContext context, bool openAI)
    {
        long started = Stopwatch.GetTimestamp();
        // One "embeddings <model>" span and one request-duration measurement per request,
        // around both the text path and the media path. Not inside the dispatcher: it
        // batches several requests into one encoder call, so a span there would belong to
        // no single request.
        using var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationEmbeddings,
            _model.ModelName, _telemetryBackend);
        try
        {
            using var document = await ReadBodyAsync(context).ConfigureAwait(false);
            JsonElement body = document.RootElement;
            ValidateModel(body);
            if (!body.TryGetProperty("input", out JsonElement input))
                throw Invalid("input is required.", "input");
            int dimensions = _model.Dimensions;
            if (body.TryGetProperty("dimensions", out JsonElement dimensionProperty)
                && dimensionProperty.ValueKind != JsonValueKind.Null)
            {
                if (dimensionProperty.ValueKind != JsonValueKind.Number || !dimensionProperty.TryGetInt32(out dimensions)
                    || dimensions <= 0 || dimensions > _model.Dimensions)
                    throw Invalid($"dimensions must be an integer between 1 and {_model.Dimensions}.", "dimensions");
            }
            string encoding = "float";
            if (openAI && body.TryGetProperty("encoding_format", out JsonElement format)
                && format.ValueKind != JsonValueKind.Null)
            {
                if (format.ValueKind != JsonValueKind.String || format.GetString() is not ("float" or "base64"))
                    throw Invalid("encoding_format must be 'float' or 'base64'.", "encoding_format");
                encoding = format.GetString()!;
            }
            bool truncate = !openAI;
            if (!openAI && body.TryGetProperty("truncate", out JsonElement truncateProperty))
            {
                if (truncateProperty.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
                    throw Invalid("truncate must be a boolean.", "truncate");
                truncate = truncateProperty.GetBoolean();
            }
            int inputCount;
            EmbeddingBatchResult result;
            if (input.ValueKind == JsonValueKind.Object || (input.ValueKind == JsonValueKind.Array &&
                input.GetArrayLength() > 0 && input[0].ValueKind == JsonValueKind.Object))
            {
                inputCount = input.ValueKind == JsonValueKind.Object ? 1 : input.GetArrayLength();
                if (telemetry.IsRecording) telemetry.SetTag(TelemetryAttributes.EmbeddingInputs, inputCount);
                result = await EmbedMediaAsync(input, context.RequestAborted).ConfigureAwait(false);
            }
            else
            {
                var inputs = ParseInputs(input, openAI, truncate, context);
                inputCount = inputs.Count;
                if (telemetry.IsRecording) telemetry.SetTag(TelemetryAttributes.EmbeddingInputs, inputCount);
                result = inputs.Count == 0 ? new EmbeddingBatchResult(Array.Empty<float[]>(), 0)
                    : await _dispatcher.EmbedAsync(inputs, context.RequestAborted).ConfigureAwait(false);
            }
            // An encoder produces no output tokens; the prompt tokens are the request's usage.
            telemetry.RecordUsage(result.PromptTokens, outputTokens: 0, cachedTokens: 0, finishReason: null,
                firstTokenAtUtc: null);
            if (result.Embeddings.Length != inputCount)
                throw new InvalidOperationException("Embedding result count does not match input count.");
            float[][] embeddings = result.Embeddings.Select(vector => ResizeAndNormalize(vector, dimensions)).ToArray();
            if (openAI)
            {
                var data = embeddings.Select((vector, index) => new { @object = "embedding", index,
                    embedding = encoding == "base64" ? (object)EncodeBase64(vector) : vector });
                await context.Response.WriteAsJsonAsync(new { @object = "list", data, model = _model.ModelName,
                    usage = new { prompt_tokens = result.PromptTokens, total_tokens = result.PromptTokens } },
                    context.RequestAborted).ConfigureAwait(false);
            }
            else
            {
                long totalNanoseconds = (long)(Stopwatch.GetElapsedTime(started).TotalMilliseconds * 1_000_000);
                await context.Response.WriteAsJsonAsync(new { model = _model.ModelName, embeddings,
                    total_duration = totalNanoseconds, load_duration = 0L, prompt_eval_count = result.PromptTokens },
                    context.RequestAborted).ConfigureAwait(false);
            }
        }
        catch (EmbeddingRequestException ex)
        {
            telemetry.MarkRefused(RefusalErrorType(ex.Status));
            await WriteErrorAsync(context, ex.Message, ex.Status, ex.Param).ConfigureAwait(false);
        }
        catch (JsonException)
        {
            telemetry.MarkRefused("invalid_json");
            await WriteErrorAsync(context, "Request body must contain valid JSON.", 400).ConfigureAwait(false);
        }
        catch (ArgumentException ex)
        {
            // Tokenization reports context overflow before native inference starts.
            telemetry.MarkRefused("invalid_input");
            await WriteErrorAsync(context, ex.Message, 400, "input").ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            // Everything else still propagates to the error middleware; the span records the
            // exception type only (a cancellation counts as cancelled, not failed).
            telemetry.MarkFailed(ex);
            throw;
        }
    }

    /// <summary>
    /// A short, stable <c>error.type</c> for a refused request. It is a metric attribute,
    /// so it names the kind of refusal, never the message.
    /// </summary>
    private static string RefusalErrorType(int status) => status switch
    {
        404 => "model_not_found",
        413 => "request_too_large",
        _ => "invalid_request",
    };

    private void ValidateModel(JsonElement body)
    {
        if (body.ValueKind != JsonValueKind.Object)
            throw Invalid("Request body must be a JSON object.");
        if (!body.TryGetProperty("model", out JsonElement requested) || requested.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(requested.GetString()))
            throw Invalid("model is required and must be a string.", "model");
        string name = requested.GetString()!;
        bool matches;
        try
        {
            matches = string.Equals(name, _model.ModelName, StringComparison.OrdinalIgnoreCase)
                || HostedModelGuard.MatchesHostedFileRequest(name, _options.StartupModelPath, allowBareModelId: true);
        }
        catch (ArgumentException)
        {
            throw Invalid("model must be a valid model identifier.", "model");
        }
        if (!matches)
            throw new EmbeddingRequestException("The requested model is not hosted by this server.", 404, "model");
    }

    private static async Task<JsonDocument> ReadBodyAsync(HttpContext context)
    {
        if (context.Request.ContentLength > MaxRequestBodyBytes)
            throw BodyTooLarge();
        // Content-Length is optional (chunked HTTP/1.1 and HTTP/2). Counting the
        // bytes read prevents JsonDocument from buffering an upload-sized body
        // before the input/token limits have a chance to reject it.
        using var limited = new BoundedReadStream(context.Request.Body);
        return await JsonDocument.ParseAsync(limited, cancellationToken: context.RequestAborted).ConfigureAwait(false);
    }

    private static EmbeddingRequestException BodyTooLarge() =>
        new($"Embedding JSON requests may contain at most {MaxRequestBodyBytes} bytes (16 MiB).", 413, null);

    private sealed class BoundedReadStream(Stream inner) : Stream
    {
        private long _bytesRead;
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        private int ReadSize(int requested) => (int)Math.Min(requested, MaxRequestBodyBytes - _bytesRead + 1);
        private int Count(int count)
        {
            _bytesRead += count;
            if (_bytesRead > MaxRequestBodyBytes) throw BodyTooLarge();
            return count;
        }
        public override int Read(byte[] buffer, int offset, int count) => Count(inner.Read(buffer, offset, ReadSize(count)));
        public override int Read(Span<byte> buffer) => Count(inner.Read(buffer[..ReadSize(buffer.Length)]));
        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default) =>
            Count(await inner.ReadAsync(buffer[..ReadSize(buffer.Length)], cancellationToken).ConfigureAwait(false));
        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        // Disposing the wrapper must leave ASP.NET's request stream open.
    }

    private List<int[]> ParseInputs(JsonElement input, bool openAI, bool truncate, HttpContext context)
    {
        var inputs = new List<int[]>();
        int totalTokens = 0;
        void AddTokens(int[] tokens)
        {
            context.RequestAborted.ThrowIfCancellationRequested();
            if (tokens.Length == 0 || tokens.Length > _model.MaxTokens)
                throw Invalid($"Each input must contain between 1 and {_model.MaxTokens} tokens.", "input");
            if (tokens.Any(token => token < 0 || token >= _model.VocabularySize))
                throw Invalid("input contains a token ID outside this model's vocabulary.", "input");
            totalTokens = checked(totalTokens + tokens.Length);
            if (inputs.Count >= MaxBatchInputs || totalTokens > MaxBatchTokens)
                throw Invalid($"A request may contain at most {MaxBatchInputs} inputs and {MaxBatchTokens} tokens.", "input");
            inputs.Add(tokens);
        }
        void AddText(JsonElement item)
        {
            if (item.ValueKind != JsonValueKind.String)
                throw Invalid("All input items must have the same type.", "input");
            string text = item.GetString()!;
            if (openAI && text.Length == 0)
                throw Invalid("Input strings must not be empty.", "input");
            if (text.Length > 1_048_576)
                throw Invalid("An input string may contain at most 1048576 characters.", "input");
            context.RequestAborted.ThrowIfCancellationRequested();
            AddTokens(_model.Tokenize(text, truncate));
        }
        int[] ReadTokens(JsonElement item)
        {
            if (item.ValueKind != JsonValueKind.Array || item.GetArrayLength() == 0 || item.GetArrayLength() > _model.MaxTokens)
                throw Invalid($"Each token input must contain between 1 and {_model.MaxTokens} token IDs.", "input");
            var tokens = new int[item.GetArrayLength()];
            int i = 0;
            foreach (JsonElement token in item.EnumerateArray())
            {
                if (token.ValueKind != JsonValueKind.Number || !token.TryGetInt32(out tokens[i++]))
                    throw Invalid("Token IDs must be integers.", "input");
            }
            return tokens;
        }

        if (input.ValueKind == JsonValueKind.String)
        {
            if (!openAI && input.GetString()!.Length == 0)
                return inputs; // Ollama's empty-input request keeps the resident model loaded.
            AddText(input);
        }
        else if (input.ValueKind == JsonValueKind.Array)
        {
            int count = input.GetArrayLength();
            if (count == 0)
            {
                if (openAI) throw Invalid("input must not be empty.", "input");
                return inputs;
            }
            if (openAI && input[0].ValueKind == JsonValueKind.Number)
                AddTokens(ReadTokens(input));
            else
            {
                if (count > MaxBatchInputs)
                    throw Invalid($"A request may contain at most {MaxBatchInputs} inputs.", "input");
                bool tokenBatch = openAI && input[0].ValueKind == JsonValueKind.Array;
                foreach (JsonElement item in input.EnumerateArray())
                {
                    if (tokenBatch) AddTokens(ReadTokens(item));
                    else AddText(item);
                }
            }
        }
        else
            throw Invalid(openAI
                ? "input must be a string, string array, token ID array, or array of token ID arrays."
                : "input must be a string or string array.", "input");
        return inputs;
    }

    internal static float[] ResizeAndNormalize(float[] vector, int dimensions)
    {
        if (vector.Length < dimensions)
            throw new InvalidOperationException("Embedding result has fewer dimensions than requested.");
        double normSquared = 0;
        foreach (float value in vector)
            if (!float.IsFinite(value))
                throw new InvalidOperationException("Embedding model returned a non-finite value.");
        var output = vector.AsSpan(0, dimensions).ToArray();
        foreach (float value in output) normSquared += (double)value * value;
        if (normSquared > 0)
        {
            double scale = 1 / Math.Sqrt(normSquared);
            for (int i = 0; i < output.Length; i++) output[i] = (float)(output[i] * scale);
        }
        return output;
    }

    internal static string EncodeBase64(float[] vector)
    {
        // Both vLLM and SGLang encode contiguous little-endian float32 values.
        var bytes = new byte[checked(vector.Length * sizeof(float))];
        for (int i = 0; i < vector.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), vector[i]);
        return Convert.ToBase64String(bytes);
    }

    public static Task WriteErrorAsync(HttpContext context, string message, int status = 400, string? param = null)
    {
        context.Response.StatusCode = status;
        object body = context.Request.Path.StartsWithSegments("/v1")
            ? new { error = new { message, type = "invalid_request_error", param,
                code = status == 404 ? "model_not_found" : (string?)null } }
            : new { error = message };
        return context.Response.WriteAsJsonAsync(body, context.RequestAborted);
    }

    private static EmbeddingRequestException Invalid(string message, string? param = null) => new(message, 400, param);
    private sealed class EmbeddingRequestException(string message, int status, string? param) : Exception(message)
    {
        public int Status { get; } = status;
        public string? Param { get; } = param;
    }
}
