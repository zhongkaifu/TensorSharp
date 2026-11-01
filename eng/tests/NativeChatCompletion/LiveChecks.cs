using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;
using TensorSharp.MicrosoftExtensionsAISample;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;
using AiMessage = Microsoft.Extensions.AI.ChatMessage;

// Explicitly opt in with an existing first GGUF shard. This test does not download weights or start a server.
internal static class LiveChecks
{
    const string Arithmetic = "What is 2 + 2? Reply with just the single digit.";
    const string Count = "Count the integers from 1 through 1000, one integer per line. Start immediately with 1 and do not stop early.";
    static int passed;
    static void Check(bool condition, string name)
    {
        if (!condition) throw new InvalidOperationException("FAIL live: " + name);
        passed++; Console.WriteLine("PASS live: " + name);
    }
    static bool Answer(string actual, string expected) => Regex.IsMatch(actual.Trim(), @"\A" + Regex.Escape(expected) + @"\.?\z");

    public static async Task RunAsync(string modelPath)
    {
        if (Path.GetFileName(modelPath) != "Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf" || !File.Exists(modelPath))
            throw new ArgumentException("This recorded GPU check requires the existing Qwen3.8 Flash Next UD-Q2_K_XL first shard and its adjacent shards.");
        Environment.SetEnvironmentVariable("MAX_CONTEXT", "4096");
        using var model = new ModelService
        {
            SchedulerConfigOverride = new SchedulerConfig
            { MaxNumRunningSequences = 1, NumBlocks = 16, BlockSize = 256, EnablePrefixCaching = true }
        };
        model.LoadModel(Path.GetFullPath(modelPath), mmProjPath: null!, backendStr: "ggml_metal");
        Check(model.LoadedBackend == "ggml_metal", "requested Metal backend loaded without a CPU backend substitution");
        Console.WriteLine($"LIVE model={model.LoadedModelName}; architecture={model.Architecture}; backend={model.LoadedBackend}; context={model.ContextTokens}");
        using IChatClient client = new TensorSharpChatClient(model);
        var history = new List<AiMessage> { new(ChatRole.User, Arithmetic) };
        var options = new ChatOptions { MaxOutputTokens = 64, Temperature = 0, Seed = 1 };
        var updates = new List<ChatResponseUpdate>();
        await foreach (var update in client.GetStreamingResponseAsync(history, options)) updates.Add(update);
        var first = updates.ToChatResponse();
        Check(Answer(first.Text, "4") && first.FinishReason == ChatFinishReason.Stop, "IChatClient streaming strict answer4 and Stop");
        Check(first.Usage is { InputTokenCount: > 0, OutputTokenCount: > 0 }, "streaming reports native token usage");
        history.AddRange(first.Messages); history.Add(new(ChatRole.User, "Multiply that answer by 3. Reply with just the number."));
        var second = await client.GetResponseAsync(history, options);
        Check(Answer(second.Text, "12") && second.FinishReason == ChatFinishReason.Stop, "same IChatClient non-streaming full-history follow-up12 and Stop");
        Check(second.AdditionalProperties?["tensorsharp.backend"] as string == "ggml_metal"
            && second.Usage is { InputTokenCount: > 0, OutputTokenCount: > 0 }, "non-streaming retains native backend and usage metrics");
        Console.WriteLine($"LIVE turn2 prompt={second.Usage?.InputTokenCount}; output={second.Usage?.OutputTokenCount}; reused={second.Usage?.CachedInputTokenCount}; native_finish={second.AdditionalProperties?["tensorsharp.finish_reason"]}");
        history.AddRange(second.Messages);
        bool rejected = false;
        try { await client.GetResponseAsync(WithUser(Arithmetic), new ChatOptions { ModelId = "other-model" }); }
        catch (ArgumentException) { rejected = true; }
        Check(rejected, "wrong model option fails before generation");

        using (var stop = new CancellationTokenSource())
        {
            bool cancelled = false; var partial = new StringBuilder();
            try
            {
                await foreach (var update in client.GetStreamingResponseAsync(WithUser(Count),
                    new ChatOptions { MaxOutputTokens = 512, Temperature = 0 }, stop.Token))
                {
                    partial.Append(update.Text);
                    if (!string.IsNullOrWhiteSpace(partial.ToString())) stop.Cancel();
                }
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested) { cancelled = true; }
            Check(cancelled && partial.ToString().TrimStart().StartsWith("1", StringComparison.Ordinal), "same adapter cancels after real partial count output");
        }
        await Recover("same adapter and loaded model recover after cancellation");
        using (var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(1)))
        {
            bool cancelled = false;
            try { await client.GetResponseAsync(WithUser(Count), new ChatOptions { MaxOutputTokens = 2048, Temperature = 0 }, deadline.Token); }
            catch (OperationCanceledException) when (deadline.IsCancellationRequested) { cancelled = true; }
            Check(cancelled, "one-second deadline cancels native generation");
        }
        await Recover("same adapter and loaded model recover after deadline");
        await foreach (var update in client.GetStreamingResponseAsync(WithUser(Count), new ChatOptions { MaxOutputTokens = 512, Temperature = 0 }))
            if (!string.IsNullOrWhiteSpace(update.Text)) break;
        await Recover("same adapter and loaded model recover after stream abandonment");
        Console.WriteLine($"{passed} real Qwen3.8 Flash Next Metal IChatClient checks passed; one ModelService and one adapter were used throughout.");

        List<AiMessage> WithUser(string text) => [.. history, new(ChatRole.User, text)];
        async Task Recover(string name)
        {
            var input = WithUser(Arithmetic);
            var response = await client.GetResponseAsync(input, options);
            Check(Answer(response.Text, "4") && ReferenceEquals(client.GetService(typeof(ModelService)), model), name);
            history = [.. input, .. response.Messages];
        }
    }
}
