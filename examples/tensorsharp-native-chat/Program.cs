using System.Globalization;
using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

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

    using var session = new ChatSession();
    var history = new List<ChatMessage>();
    await SendAsync(options.Prompt);
    if (options.FollowUp is not null) await SendAsync(options.FollowUp);
    return 0;

    async Task SendAsync(string prompt)
    {
        history.Add(new ChatMessage { Role = "user", Content = prompt });
        var parser = OutputParserFactory.Create(service.Architecture);
        parser.Init(enableThinking: false, tools: null);
        var answer = new StringBuilder();
        ChatStreamUpdate? terminal = null;
        // The deadline covers generation, not synchronous model loading.
        cancellation.CancelAfter(TimeSpan.FromSeconds(options.TimeoutSeconds));
        await foreach (var update in service.ChatStreamWithMetricsAsync(
            session, history, options.MaxTokens, cancellation.Token,
            new SamplingConfig { Temperature = 0, Seed = 1, RepetitionPenalty = 1 }, enableThinking: false))
        {
            if (!update.Done && update.RawGenerationSuffix is { } suffix) parser.SetGenerationPromptSuffix(suffix);
            // TensorSharp knows each family's output protocol; do not print reasoning markup as the answer.
            string text = (update.IsParsed ? update.Piece : parser.Add(update.Piece ?? "", update.Done).Content) ?? "";
            Console.Write(text);
            answer.Append(text);
            if (update.Done) terminal = update;
        }
        cancellation.CancelAfter(Timeout.InfiniteTimeSpan);
        Console.WriteLine();
        if (terminal is not { } done) throw new InvalidOperationException("TensorSharp ended the stream without terminal metrics. Output may be partial.");
        Console.Error.WriteLine($"[TensorSharp native] finish={done.FinishReason}; prompt_tokens={done.PromptTokens}; generated_tokens={done.EvalTokens}; kv_cache_reused_tokens={done.KvCacheReusedTokens}");
        if (done.FinishReason is "error" or "aborted" or "cancelled")
            throw new InvalidOperationException($"TensorSharp generation stopped with {done.FinishReason}. Output may be partial.");
        if (string.IsNullOrWhiteSpace(answer.ToString())) throw new InvalidOperationException("TensorSharp returned no answer text. Check model support and the token budget.");
        history.Add(new ChatMessage
        {
            Role = "assistant", Content = answer.ToString(),
            // Preserve TensorSharp's exact generated-token boundary for the next turn's KV reuse.
            RawOutputTokens = done.RawOutputTokens?.ToList(),
            RawPromptTrailingWhitespace = done.RawPromptTrailingWhitespace,
            RawGenerationSuffix = done.RawGenerationSuffix
        });
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

sealed record Options(string Model, string Backend, string Prompt, string? FollowUp,
    int MaxTokens, int ContextTokens, int TimeoutSeconds, bool Help)
{
    public const string Usage = """
        TensorSharp direct .NET runtime chat (.NET 10; existing local GGUF required)
        dotnet run --project examples/tensorsharp-native-chat -- [options]
          --model PATH            Existing GGUF; or TENSORSHARP_MODEL_PATH
          --backend NAME          ggml_cpu (default), ggml_metal, ggml_cuda, ggml_vulkan
          --prompt TEXT           Default: Answer briefly: what is 2 + 2?
          --follow-up TEXT        Optional second turn in the same ChatSession
          --max-tokens N          Default: 128
          --context-tokens N      Default: 2048; sets this process's MAX_CONTEXT
          --timeout-seconds N     Per generation, after model loading; default: 120
          --help                  Show this help
        Ctrl+C cancels generation. Model loading is synchronous and not cancellable through this API.
        """;

    public static Options Parse(string[] args)
    {
        string? model = Environment.GetEnvironmentVariable("TENSORSHARP_MODEL_PATH");
        string backend = "ggml_cpu", prompt = "Answer briefly: what is 2 + 2?";
        string? followUp = null;
        int maxTokens = 128, contextTokens = 2048, timeout = 120;
        if (args.Contains("--help")) return new("", backend, prompt, null, maxTokens, contextTokens, timeout, true);
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
                case "--max-tokens": maxTokens = Number(Value(), option); break;
                case "--context-tokens": contextTokens = Number(Value(), option); break;
                case "--timeout-seconds": timeout = Number(Value(), option); break;
                default: throw new ArgumentException($"Unknown option {option}. Use --help.");
            }
        }
        if (string.IsNullOrWhiteSpace(model)) throw new ArgumentException("Pass --model or set TENSORSHARP_MODEL_PATH to an existing supported GGUF file.");
        if (backend is not ("ggml_cpu" or "ggml_metal" or "ggml_cuda" or "ggml_vulkan"))
            throw new ArgumentException("--backend must be ggml_cpu, ggml_metal, ggml_cuda or ggml_vulkan.");
        if (string.IsNullOrWhiteSpace(prompt) || followUp is not null && string.IsNullOrWhiteSpace(followUp))
            throw new ArgumentException("Prompts must not be empty.");
        if (contextTokens < 1024 || maxTokens >= contextTokens - 1024)
            throw new ArgumentException("Choose --context-tokens above 1024 and reserve more than --max-tokens + 1024 tokens for the chat prompt.");
        return new(Path.GetFullPath(model), backend, prompt, followUp, maxTokens, contextTokens, timeout, false);
    }

    static int Number(string value, string name) => int.TryParse(value, out int n) && n is > 0 and <= 65536
        ? n : throw new ArgumentException($"{name} must be an integer from 1 to 65536.");
}
