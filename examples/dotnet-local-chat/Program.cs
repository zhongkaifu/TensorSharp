using System.ClientModel;
using System.ClientModel.Primitives;
using System.Net.Http.Headers;
using System.Text.Json;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;

return await RunAsync(args);

static async Task<int> RunAsync(string[] args)
{
    using var cancelled = new CancellationTokenSource();
    Console.CancelKeyPress += (_, e) => { e.Cancel = true; cancelled.Cancel(); };
    try
    {
        var settings = Settings.Parse(args);
        if (settings.Help) { Console.WriteLine(Settings.Usage); return 0; }
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancelled.Token);
        deadline.CancelAfter(TimeSpan.FromSeconds(settings.TimeoutSeconds));
        var ct = deadline.Token;
        // One transport for discovery and both SDK paths. Do not follow redirects with credentials.
        using var http = new HttpClient(new HttpClientHandler { AllowAutoRedirect = false })
        { Timeout = Timeout.InfiniteTimeSpan };
        string key = Environment.GetEnvironmentVariable("TENSORSHARP_API_KEY") ?? "local-not-a-secret";
        var models = await DiscoverModelsAsync(http, settings.Endpoint, key, ct);
        if (settings.ListModels)
        {
            foreach (string id in models) Console.WriteLine(id);
            if (models.Length == 0) throw new InvalidOperationException(
                "No models advertised by /v1/models. Start TensorSharp with --model pointing to an existing supported model.");
            return 0;
        }
        string model = settings.Model ?? (models.Length == 1 ? models[0] : throw new InvalidOperationException(
            models.Length == 0 ? "No models advertised. Start TensorSharp with --model pointing to an existing supported model."
            : "Multiple models advertised. Choose an exact ID with --model; use --list-models to see IDs."));
        if (!models.Contains(model, StringComparer.Ordinal)) throw new InvalidOperationException(
            "The requested model is not advertised by /v1/models. Use --list-models and pass an exact ID with --model.");

        Console.Error.WriteLine($"Streaming with {settings.Client}; model={model}; endpoint={settings.Endpoint}");
        var sdk = new OpenAI.Chat.ChatClient(model, new ApiKeyCredential(key), new OpenAIClientOptions
        {
            Endpoint = settings.Endpoint,
            Transport = new HttpClientPipelineTransport(http),
            RetryPolicy = new ClientRetryPolicy(0),
            NetworkTimeout = TimeSpan.FromSeconds(settings.TimeoutSeconds)
        });
        bool finished = false;
        bool limited = false;
        bool hasText = false;
        if (settings.Client == "extensions")
        {
            // Microsoft's official adapter wraps the official OpenAI client. No TensorSharp adapter required.
            using IChatClient chat = sdk.AsIChatClient();
            await foreach (var update in chat.GetStreamingResponseAsync(
                [new Microsoft.Extensions.AI.ChatMessage(ChatRole.User, settings.Prompt)],
                new ChatOptions { MaxOutputTokens = settings.MaxTokens }, ct))
            {
                Console.Write(update.Text);
                hasText |= !string.IsNullOrWhiteSpace(update.Text);
                finished |= update.FinishReason is not null;
                limited |= update.FinishReason == Microsoft.Extensions.AI.ChatFinishReason.Length;
            }
        }
        else
        {
            await foreach (var update in sdk.CompleteChatStreamingAsync(
                [new UserChatMessage(settings.Prompt)],
                new ChatCompletionOptions { MaxOutputTokenCount = settings.MaxTokens }, ct))
            {
                foreach (var part in update.ContentUpdate)
                {
                    Console.Write(part.Text);
                    hasText |= !string.IsNullOrWhiteSpace(part.Text);
                }
                finished |= update.FinishReason is not null;
                limited |= update.FinishReason == OpenAI.Chat.ChatFinishReason.Length;
            }
        }
        Console.WriteLine();
        if (!finished) throw new InvalidOperationException(
            "The stream ended without a finish_reason. Output may be partial; inspect the TensorSharp server log.");
        if (!hasText) throw new InvalidOperationException(
            "The stream completed without assistant text. Check model compatibility, token limits, and the TensorSharp server log.");
        if (limited) Console.Error.WriteLine("Generation reached the token limit. Increase --max-tokens and the server's cap if needed.");
        return 0;
    }
    catch (OperationCanceledException)
    {
        Console.Error.WriteLine(cancelled.IsCancellationRequested ? "Cancelled. Output may be partial."
            : "Request timed out. Check model loading/server logs or increase --timeout-seconds. Output may be partial.");
        return cancelled.IsCancellationRequested ? 130 : 1;
    }
    catch (ClientResultException e)
    {
        Console.Error.WriteLine($"Chat request failed (HTTP {e.Status}). {HttpHint(e.Status)}\n{e.Message}");
        return 1;
    }
    catch (HttpRequestException e)
    {
        Console.Error.WriteLine($"Cannot reach the configured TensorSharp endpoint. Start the server, check --base-url and /v1/models. {e.Message}");
        return 1;
    }
    catch (JsonException)
    {
        Console.Error.WriteLine("The server returned invalid JSON or streaming data. Check --base-url and the TensorSharp server log.");
        return 1;
    }
    catch (Exception e) when (e is ArgumentException or InvalidOperationException or FormatException or IOException)
    {
        Console.Error.WriteLine($"Error: {e.Message}");
        return 1;
    }
}

static async Task<string[]> DiscoverModelsAsync(HttpClient http, Uri endpoint, string key, CancellationToken ct)
{
    using var request = new HttpRequestMessage(HttpMethod.Get, new Uri(endpoint, "models"));
    request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
    using var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct);
    if (!response.IsSuccessStatusCode) throw new InvalidOperationException(
        $"Model discovery failed (HTTP {(int)response.StatusCode}) at /v1/models. {HttpHint((int)response.StatusCode)}");
    using var json = await JsonDocument.ParseAsync(await response.Content.ReadAsStreamAsync(ct), cancellationToken: ct);
    if (!json.RootElement.TryGetProperty("data", out var data) || data.ValueKind != JsonValueKind.Array)
        throw new InvalidOperationException("Invalid /v1/models response: expected a JSON data array.");
    var ids = new List<string>();
    foreach (var item in data.EnumerateArray())
    {
        if (!item.TryGetProperty("id", out var id) || id.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(id.GetString()))
            throw new InvalidOperationException("Invalid /v1/models response: each model needs a nonempty string id.");
        ids.Add(id.GetString()!);
    }
    return ids.Distinct(StringComparer.Ordinal).ToArray();
}

static string HttpHint(int status) => status switch
{
    401 or 403 => "Check TENSORSHARP_API_KEY and any authenticated proxy configuration.",
    404 => "The base URL must end in /v1 and point at the TensorSharp OpenAI-compatible server.",
    400 or 422 => "Check the model ID, request options, and server log.",
    >= 300 and < 400 => "Redirects are disabled. Set --base-url to the final trusted endpoint.",
    >= 500 => "Check model availability and the TensorSharp server log.",
    _ => "Check the TensorSharp server log."
};

sealed record Settings(Uri Endpoint, string? Model, string Prompt, string Client, int MaxTokens,
    int TimeoutSeconds, bool ListModels, bool Help)
{
    public const string Usage = """
        TensorSharp one-turn streaming chat (.NET 10)
        dotnet run -- [options]
          --list-models             Discover IDs using GET /v1/models
          --prompt TEXT             User message (default: a short introduction)
          --model ID                Exact advertised ID (auto-select if exactly one)
          --client extensions|openai Default: extensions (IChatClient)
          --base-url URL            Default: http://127.0.0.1:8080/v1
          --max-tokens N            Default: 128
          --timeout-seconds N       Total discovery + generation deadline; default: 120
          --allow-remote            Allow an explicitly chosen HTTPS remote endpoint
          --help                    Show this help
        Environment: TENSORSHARP_BASE_URL, TENSORSHARP_MODEL, TENSORSHARP_API_KEY
        CLI overrides environment. No OpenAI account or real API key needed for local TensorSharp.
        """;

    public static Settings Parse(string[] args)
    {
        string baseUrl = Environment.GetEnvironmentVariable("TENSORSHARP_BASE_URL") ?? "http://127.0.0.1:8080/v1";
        string? model = Environment.GetEnvironmentVariable("TENSORSHARP_MODEL");
        string prompt = "Explain what a local LLM inference server does in one sentence.";
        string client = "extensions";
        int maxTokens = 128, timeoutSeconds = 120;
        bool list = false, remote = false;
        if (args.Contains("--help")) return new(new Uri("http://127.0.0.1:8080/v1/"), null, prompt, client, maxTokens, timeoutSeconds, false, true);
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            string Value() => ++i < args.Length ? args[i] : throw new ArgumentException($"{option} requires a value. Use --help.");
            switch (option)
            {
                case "--base-url": baseUrl = Value(); break;
                case "--model": model = Value(); break;
                case "--prompt": prompt = Value(); break;
                case "--client": client = Value(); break;
                case "--max-tokens": maxTokens = PositiveInt(Value(), option); break;
                case "--timeout-seconds": timeoutSeconds = PositiveInt(Value(), option); break;
                case "--list-models": list = true; break;
                case "--allow-remote": remote = true; break;
                default: throw new ArgumentException($"Unknown option {option}. Use --help.");
            }
        }
        if (client is not ("extensions" or "openai")) throw new ArgumentException("--client must be extensions or openai.");
        if (string.IsNullOrWhiteSpace(prompt)) throw new ArgumentException("--prompt must not be empty.");
        if (!Uri.TryCreate(baseUrl, UriKind.Absolute, out var uri) || uri.Scheme is not ("http" or "https")
            || uri.UserInfo.Length != 0 || uri.Query.Length != 0 || uri.Fragment.Length != 0
            || !uri.AbsolutePath.TrimEnd('/').EndsWith("/v1", StringComparison.Ordinal))
            throw new ArgumentException("--base-url must be an HTTP(S) URL ending in /v1, without credentials, query, or fragment.");
        if (!uri.IsLoopback && (!remote || uri.Scheme != "https")) throw new ArgumentException(
            "Use a loopback endpoint by default. Remote endpoints require HTTPS and explicit --allow-remote; add authentication at a trusted proxy.");
        return new(new Uri(uri.AbsoluteUri.TrimEnd('/') + "/"), string.IsNullOrWhiteSpace(model) ? null : model,
            prompt, client, maxTokens, timeoutSeconds, list, false);
    }

    static int PositiveInt(string value, string option) => int.TryParse(value, out int n) && n is > 0 and <= 86400
        ? n : throw new ArgumentException($"{option} must be an integer from 1 to 86400.");
}
