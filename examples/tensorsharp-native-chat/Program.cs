using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.AI;
using TensorSharp.MicrosoftExtensionsAISample;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;
using AiMessage = Microsoft.Extensions.AI.ChatMessage;

using var cancellation = new CancellationTokenSource();
bool interrupted = false;
Console.CancelKeyPress += (_, e) => { e.Cancel = true; interrupted = true; cancellation.Cancel(); };
try
{
    var options = Options.Parse(args);
    if (options.Help) { Console.WriteLine(Options.Usage); return 0; }
    if (!File.Exists(options.Model)) throw new ArgumentException("Model file not found. Pass --model with an existing supported GGUF file.");

    // These settings belong to this console process. Nothing is downloaded or hosted.
    Environment.SetEnvironmentVariable("MAX_CONTEXT", options.ContextTokens.ToString(CultureInfo.InvariantCulture));
    using var service = new ModelService
    {
        SchedulerConfigOverride = new SchedulerConfig
        {
            MaxNumRunningSequences = 1,
            NumBlocks = (options.ContextTokens + 255) / 256,
            BlockSize = 256,
            EnablePrefixCaching = true
        }
    };
    Console.Error.WriteLine("[TensorSharp native] Loading local weights through TensorSharp.Chat...");
    // The API is public and host-neutral despite its retained TensorSharp.Server namespace.
    service.LoadModel(options.Model, mmProjPath: null!, backendStr: options.Backend);
    cancellation.Token.ThrowIfCancellationRequested();
    Console.Error.WriteLine($"[TensorSharp native] model={service.LoadedModelName}; architecture={service.Architecture}; backend={service.LoadedBackend}; context_tokens={service.ContextTokens}");

    if (!string.Equals(service.LoadedBackend, options.Backend, StringComparison.Ordinal))
        throw new InvalidOperationException($"Requested backend {options.Backend}, but TensorSharp loaded {service.LoadedBackend}. Refusing a silent backend substitution.");
    using IChatClient chat = new TensorSharpChatClient(service);
    var history = new List<AiMessage>();
    int completedTurns = 0;
    await SendAsync(options.Prompt, stream: options.Mode != "response");
    if (options.FollowUp is not null) await SendAsync(options.FollowUp, stream: options.Mode == "stream");
    return 0;

    async Task SendAsync(string prompt, bool stream)
    {
        history.Add(new AiMessage(ChatRole.User, prompt));
        var settings = new ChatOptions { MaxOutputTokens = options.MaxTokens, Temperature = 0, Seed = 1 };
        ChatResponse response;
        cancellation.CancelAfter(TimeSpan.FromSeconds(options.TimeoutSeconds));
        if (stream)
        {
            AdditionalPropertiesDictionary? metrics = null;
            async IAsyncEnumerable<ChatResponseUpdate> PrintAsync()
            {
                await foreach (var update in chat.GetStreamingResponseAsync(history, settings, cancellation.Token))
                {
                    Console.Write(update.Text);
                    metrics = update.AdditionalProperties ?? metrics;
                    yield return update;
                }
            }
            response = await PrintAsync().ToChatResponseAsync(cancellation.Token);
            response.AdditionalProperties = metrics;
        }
        else
        {
            response = await chat.GetResponseAsync(history, settings, cancellation.Token);
            Console.Write(response.Text);
        }
        cancellation.Token.ThrowIfCancellationRequested();
        cancellation.CancelAfter(Timeout.InfiniteTimeSpan);
        Console.WriteLine();
        Console.Error.WriteLine($"[TensorSharp IChatClient] mode={(stream ? "stream" : "response")}; finish={response.FinishReason}; native_finish={response.AdditionalProperties?["tensorsharp.finish_reason"]}; prompt_tokens={response.Usage?.InputTokenCount}; generated_tokens={response.Usage?.OutputTokenCount}; kv_cache_reused_tokens={response.Usage?.CachedInputTokenCount}");
        if (options.RegressionJson)
            Console.Error.WriteLine("[TensorSharp regression] " + JsonSerializer.Serialize(new
            {
                schema_version = 1, turn = ++completedTurns, text = response.Text,
                model = service.LoadedModelName, architecture = service.Architecture, backend = service.LoadedBackend,
                mode = stream ? "stream" : "response", finish = response.FinishReason?.ToString(),
                native_finish = response.AdditionalProperties?["tensorsharp.finish_reason"],
                prompt_tokens = response.Usage?.InputTokenCount, generated_tokens = response.Usage?.OutputTokenCount,
                kv_cache_reused_tokens = response.Usage?.CachedInputTokenCount
            }));
        history.AddRange(response.Messages);
    }
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine(interrupted ? "TensorSharp generation cancelled. Output may be partial."
        : "TensorSharp generation timed out. Increase --timeout-seconds if needed. Output may be partial.");
    return interrupted ? 130 : 1;
}
catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
{
    Console.Error.WriteLine("TensorSharp native backend could not load. Build the matching native backend for this checkout and architecture; see the README. " + e.Message);
    return 1;
}
catch (Exception e) when (e is ArgumentException or InvalidOperationException or IOException or NotSupportedException)
{
    Console.Error.WriteLine("TensorSharp native chat failed: " + e.Message);
    return 1;
}

internal static class NativeCompletion
{
    public static ChatStreamUpdate Require(ChatStreamUpdate? terminal, bool hasText, CancellationToken cancellationToken)
    {
        // Cancellation may arrive as a terminal update instead of an exception from the iterator.
        // Preserve the caller's cancellation/deadline path before interpreting finish reasons.
        cancellationToken.ThrowIfCancellationRequested();
        if (terminal is not { } done) throw new InvalidOperationException("TensorSharp ended the stream without terminal metrics. Output may be partial.");
        if (done.FinishReason is "error" or "aborted" or "cancelled")
            throw new InvalidOperationException($"TensorSharp generation stopped with {done.FinishReason}. Output may be partial.");
        if (!hasText) throw new InvalidOperationException("TensorSharp returned no answer text. Check model support and the token budget.");
        return done;
    }
}

sealed record Options(string Model, string Backend, string Prompt, string? FollowUp,
    int MaxTokens, int ContextTokens, int TimeoutSeconds, string Mode, bool Help, bool RegressionJson)
{
    public const string Usage = """
        Microsoft.Extensions.AI over TensorSharp's direct .NET runtime (.NET 10)
        dotnet run --project examples/tensorsharp-native-chat -- [options]
          --model PATH            Existing GGUF; or TENSORSHARP_MODEL_PATH
          --backend NAME          ggml_metal (default), ggml_cuda, ggml_vulkan, ggml_cpu
          --prompt TEXT           Default: Answer briefly: what is 2 + 2?
          --follow-up TEXT        Optional second turn through the same IChatClient
          --mode mixed|stream|response  Default: streaming first turn, non-streaming follow-up
          --max-tokens N          Default: 128
          --context-tokens N      Default: 4096; sets this process's MAX_CONTEXT
          --timeout-seconds N     Per generation, after model loading; default: 120
          --regression-json       Emit complete per-turn diagnostic JSON to stderr (may contain prompt-derived text)
          --help                  Show this help
        Ctrl+C cancels generation. Model loading is synchronous and not cancellable through this API.
        """;

    public static Options Parse(string[] args)
    {
        string? model = Environment.GetEnvironmentVariable("TENSORSHARP_MODEL_PATH");
        string backend = "ggml_metal", prompt = "What is 2 + 2? Reply with just the single digit.", mode = "mixed";
        string? followUp = null;
        int maxTokens = 128, contextTokens = 4096, timeout = 120;
        bool regressionJson = false;
        if (args.Contains("--help")) return new("", backend, prompt, null, maxTokens, contextTokens, timeout, mode, true, false);
        for (int i = 0; i < args.Length; i++)
        {
            string option = args[i];
            string Value() => ++i < args.Length ? args[i] : throw new ArgumentException($"{option} requires a value. Use --help.");
            switch (option)
            {
                case "--model": model = Value(); break;
                case "--backend": backend = Value(); break;
                case "--prompt": prompt = Value(); break;
                case "--follow-up": followUp = Value(); break;
                case "--mode": mode = Value(); break;
                case "--max-tokens": maxTokens = Number(Value(), option); break;
                case "--context-tokens": contextTokens = Number(Value(), option); break;
                case "--timeout-seconds": timeout = Number(Value(), option); break;
                case "--regression-json": regressionJson = true; break;
                default: throw new ArgumentException($"Unknown option {option}. Use --help.");
            }
        }
        if (string.IsNullOrWhiteSpace(model)) throw new ArgumentException("Pass --model or set TENSORSHARP_MODEL_PATH to an existing supported GGUF file.");
        if (backend is not ("ggml_cpu" or "ggml_metal" or "ggml_cuda" or "ggml_vulkan"))
            throw new ArgumentException("--backend must be ggml_cpu, ggml_metal, ggml_cuda or ggml_vulkan.");
        if (mode is not ("mixed" or "stream" or "response")) throw new ArgumentException("--mode must be mixed, stream or response.");
        if (string.IsNullOrWhiteSpace(prompt) || followUp is not null && string.IsNullOrWhiteSpace(followUp))
            throw new ArgumentException("Prompts must not be empty.");
        if (contextTokens < 1024 || maxTokens >= contextTokens - 1024)
            throw new ArgumentException("Choose --context-tokens above 1024 and reserve more than --max-tokens + 1024 tokens for the chat prompt.");
        return new(Path.GetFullPath(model), backend, prompt, followUp, maxTokens, contextTokens, timeout, mode, false, regressionJson);
    }

    static int Number(string value, string name) => int.TryParse(value, out int n) && n is > 0 and <= 65536
        ? n : throw new ArgumentException($"{name} must be an integer from 1 to 65536.");
}
