using System.Reflection;
using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using TensorSharp.MicrosoftExtensionsAISample;
using TensorSharp.Runtime;
using TensorSharp.Server;
using AiMessage = Microsoft.Extensions.AI.ChatMessage;
using NativeMessage = TensorSharp.Runtime.ChatMessage;
using Generator = System.Func<TensorSharp.Server.ChatSession, System.Collections.Generic.List<TensorSharp.Runtime.ChatMessage>, int, System.Threading.CancellationToken, TensorSharp.Runtime.SamplingConfig, System.Collections.Generic.IAsyncEnumerable<TensorSharp.Server.ChatStreamUpdate>>;

internal static class AdapterChecks
{
    static int passed;
    static TensorSharpChatClient Client(Generator generate, SemaphoreSlim? gate = null) =>
        (TensorSharpChatClient)typeof(TensorSharpChatClient).GetConstructors(BindingFlags.NonPublic | BindingFlags.Instance)
            .Single().Invoke(["fixture.gguf", "gemma4", "ggml_metal", generate, gate ?? new SemaphoreSlim(1, 1), null]);
    static void Check(bool condition, string name)
    {
        if (!condition) throw new InvalidOperationException("FAIL " + name);
        passed++;
        Console.WriteLine("PASS adapter: " + name);
    }
    static async Task Reject<T>(Func<Task> action, string name) where T : Exception
    {
        try { await action(); }
        catch (T) { Check(true, name); return; }
        throw new InvalidOperationException("FAIL expected " + typeof(T).Name + ": " + name);
    }
    static NativeMessage Copy(NativeMessage message) => new() { Role = message.Role, Content = message.Content };
    static ChatStreamUpdate Done(string reason = "eos", int reuse = 8) => new("", true, 32, 2, reuse, 100, 60, 40, reason);
    static async IAsyncEnumerable<ChatStreamUpdate> Answer(string answer, string finish = "eos",
        [EnumeratorCancellation] CancellationToken token = default)
    {
        await Task.Yield(); token.ThrowIfCancellationRequested();
        yield return ChatStreamUpdate.Text(answer);
        yield return Done(finish);
    }

    public static async Task RunAsync()
    {
        var sessions = new List<ChatSession>();
        var inputs = new List<List<NativeMessage>>();
        int capturedLimit = 0;
        SamplingConfig? capturedSampling = null;
        IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            CancellationToken token, SamplingConfig sampling)
        {
            sessions.Add(session); inputs.Add(history.Select(Copy).ToList()); capturedLimit = limit; capturedSampling = sampling;
            return Answer(inputs.Count == 1 ? "4" : "12", token: token);
        }
        using (var client = Client(Generate))
        {
            var history = new List<AiMessage> { new(ChatRole.User, "2 + 2?") };
            var options = new ChatOptions { MaxOutputTokens = 17, Temperature = 0.25f, TopP = 0.75f, TopK = 3,
                FrequencyPenalty = 0.1f, PresencePenalty = 0.2f, Seed = 7, StopSequences = ["END"] };
            var response = await client.GetResponseAsync(history, options);
            Check(response.Text == "4" && response.FinishReason == ChatFinishReason.Stop, "non-streaming official aggregation");
            Check(response.Usage is { InputTokenCount: 32, OutputTokenCount: 2, TotalTokenCount: 34, CachedInputTokenCount: 8 }, "standard usage includes cached tokens within input");
            Check(response.AdditionalProperties?["tensorsharp.backend"] as string == "ggml_metal"
                && response.AdditionalProperties?["tensorsharp.finish_reason"] as string == "eos", "native metrics survive aggregation");
            Check(response.ConversationId is null && response.ModelId == "fixture.gguf", "full-history contract returns no conversation ID");
            Check(capturedLimit == 17 && capturedSampling is { Temperature: 0.25f, TopP: 0.75f, TopK: 3,
                FrequencyPenalty: 0.1f, PresencePenalty: 0.2f, Seed: 7 }, "sampling options map to native values");
            capturedSampling!.StopSequences!.Add("MUTATED");
            Check(options.StopSequences!.Count == 1 && history.Count == 1 && history[0].Text == "2 + 2?", "caller history and options are not mutated");
            history.AddRange(response.Messages); history.Add(new(ChatRole.User, "Multiply that by 3."));
            var updates = new List<ChatResponseUpdate>();
            await foreach (var update in client.GetStreamingResponseAsync(history)) updates.Add(update);
            Check(string.Concat(updates.Select(update => update.Text)) == "12", "streaming follow-up returns content");
            Check(sessions[0] == sessions[1] && inputs[1].Count == 3 && inputs[1][1].Content == "4", "same native session receives full exact history");
            Check(updates.Select(update => update.MessageId).Distinct().Count() == 1
                && updates[0].MessageId != response.Messages[0].MessageId, "message ID stable per response and unique across calls");
            var metadata = (ChatClientMetadata)client.GetService(typeof(ChatClientMetadata))!;
            Check(metadata.ProviderName == "tensorsharp" && metadata.ProviderUri is null
                && ReferenceEquals(client.GetService(typeof(IChatClient)), client), "GetService reports native provider and client");
            Check(client.GetService(typeof(string)) is null && client.GetService(typeof(IChatClient), "unknown") is null, "unknown services and keys return null");
            await Reject<ArgumentException>(() => client.GetResponseAsync([new(ChatRole.User, "unrelated")]), "different conversation rejected before native call");
        }
        Check(sessions.All(session => session.IsDisposed), "adapter disposal releases its native session");

        using (var validation = Client((_, _, _, token, _) => Answer("4", token: token)))
        {
            Task Request(ChatOptions options) => validation.GetResponseAsync([new(ChatRole.User, "q")], options);
            await Reject<NotSupportedException>(() => Request(new() { ConversationId = "session" }), "delta-only conversation IDs rejected");
            await Reject<NotSupportedException>(() => Request(new() { AdditionalProperties = new() { ["unknown"] = true } }), "unsupported provider options rejected");
            await Reject<NotSupportedException>(() => Request(new() { ToolMode = ChatToolMode.Auto }), "unsupported tools rejected");
            await Reject<ArgumentException>(() => Request(new() { ModelId = "other" }), "model switching rejected");
            await Reject<ArgumentOutOfRangeException>(() => Request(new() { Seed = (long)int.MaxValue + 1 }), "64-bit seed overflow rejected");
            await Reject<ArgumentOutOfRangeException>(() => Request(new() { MaxOutputTokens = 0 }), "invalid output limit rejected");
            await Reject<ArgumentOutOfRangeException>(() => Request(new() { Temperature = float.NaN }), "nonfinite sampling rejected");
            await Reject<NotSupportedException>(() => validation.GetResponseAsync([new(ChatRole.Tool, "q")]), "unsupported message role rejected");
            await Reject<NotSupportedException>(() => validation.GetResponseAsync([new(ChatRole.User,
                [new DataContent(new byte[] { 1 }, "image/png")])]), "unsupported media rejected");
            await Reject<NotSupportedException>(() => validation.GetResponseAsync([new(ChatRole.User, "q")
                { AdditionalProperties = new() { ["hidden_instruction"] = true } }]), "unsupported message semantic properties rejected");
            await Reject<NotSupportedException>(() => validation.GetResponseAsync([new(ChatRole.User, "q")
                { RawRepresentation = new object() }]), "unsupported raw message representation rejected");
        }
        foreach (string finish in new[] { "max_tokens", "thinking_budget", "repetition" })
        {
            using var client = Client((_, _, _, token, _) => Answer("4", finish, token));
            Check((await client.GetResponseAsync([new(ChatRole.User, "q")])).FinishReason == ChatFinishReason.Length, finish + " maps to Length");
        }
        await CancellationAndRecoveryAsync();
        await ParsingAsync();
        await AbandonmentAsync();
        await SharedGateAsync();
        await DisposeWhileModelBusyAsync();
        Console.WriteLine($"{passed} native IChatClient adapter checks passed; no model or native backend loaded.");
    }

    static async Task ParsingAsync()
    {
        async IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            [EnumeratorCancellation] CancellationToken token, SamplingConfig sampling)
        {
            await Task.Yield();
            if (history[^1].Content == "primed")
            {
                yield return ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" };
                yield return ChatStreamUpdate.Text("hidden reasoning<channel|>4");
                yield return Done();
            }
            else
            {
                yield return ChatStreamUpdate.Text("<|channel>thought\nliteral") with { IsParsed = true };
                yield return Done() with { IsParsed = true };
            }
        }
        using var primed = Client(Generate);
        Check((await primed.GetResponseAsync([new(ChatRole.User, "primed")])).Text == "4", "generation suffix primes family parser and Done flushes answer");
        using var parsed = Client(Generate);
        Check((await parsed.GetResponseAsync([new(ChatRole.User, "already parsed")])).Text == "<|channel>thought\nliteral", "already-parsed native content is not parsed twice");
    }

    static async Task CancellationAndRecoveryAsync()
    {
        using var cancel = new CancellationTokenSource();
        var sessions = new List<ChatSession>();
        async IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            [EnumeratorCancellation] CancellationToken token, SamplingConfig sampling)
        {
            sessions.Add(session); await Task.Yield();
            yield return ChatStreamUpdate.Text("4");
            if (history[^1].Content == "cancel") cancel.Cancel();
            yield return Done(history[^1].Content == "cancel" ? "cancelled" : "eos");
        }
        using var client = Client(Generate);
        await Reject<OperationCanceledException>(() => client.GetResponseAsync([new(ChatRole.User, "cancel")], cancellationToken: cancel.Token), "cancelled terminal preserves caller cancellation");
        Check((await client.GetResponseAsync([new(ChatRole.User, "recover")])).Text == "4", "same adapter recovers after cancellation");
        Check(sessions.Count == 2 && sessions[0].IsDisposed && sessions[0] != sessions[1], "failed session retired without replacing adapter or model");
    }

    static async Task AbandonmentAsync()
    {
        bool stopped = false;
        int calls = 0;
        async IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            [EnumeratorCancellation] CancellationToken token, SamplingConfig sampling)
        {
            calls++;
            try
            {
                yield return ChatStreamUpdate.Text("4");
                if (calls == 1) await Task.Delay(Timeout.Infinite, token);
                yield return Done();
            }
            finally { stopped = true; }
        }
        using var client = Client(Generate);
        await foreach (var update in client.GetStreamingResponseAsync([new(ChatRole.User, "abandon")])) break;
        Check(stopped, "abandoned stream cancels and disposes generator before returning");
        Check((await client.GetResponseAsync([new(ChatRole.User, "recover")])).Text == "4", "same adapter recovers after stream abandonment");
        client.Dispose();
        await Reject<ObjectDisposedException>(() => client.GetResponseAsync([new(ChatRole.User, "disposed")]), "disposed client rejects requests");
    }

    static async Task SharedGateAsync()
    {
        int active = 0, maximum = 0;
        var sessions = new List<ChatSession>();
        async IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            [EnumeratorCancellation] CancellationToken token, SamplingConfig sampling)
        {
            sessions.Add(session); int current = Interlocked.Increment(ref active); maximum = Math.Max(maximum, current);
            try { await Task.Delay(10, token); yield return ChatStreamUpdate.Text("4"); yield return Done(); }
            finally { Interlocked.Decrement(ref active); }
        }
        var gate = new SemaphoreSlim(1, 1);
        using var first = Client(Generate, gate);
        using var second = Client(Generate, gate);
        await Task.WhenAll(first.GetResponseAsync([new(ChatRole.User, "first")]), second.GetResponseAsync([new(ChatRole.User, "second")]));
        Check(maximum == 1 && sessions.Count == 2 && sessions[0] != sessions[1], "shared model gate serializes distinct conversation sessions");
    }

    static async Task DisposeWhileModelBusyAsync()
    {
        bool stopped = false;
        var sessions = new List<ChatSession>();
        async IAsyncEnumerable<ChatStreamUpdate> Generate(ChatSession session, List<NativeMessage> history, int limit,
            [EnumeratorCancellation] CancellationToken token, SamplingConfig sampling)
        {
            sessions.Add(session);
            try
            {
                yield return ChatStreamUpdate.Text("4");
                if (history[^1].Content == "pause") await Task.Delay(Timeout.Infinite, token);
                yield return Done();
            }
            finally { stopped = true; }
        }
        var gate = new SemaphoreSlim(1, 1);
        using var active = Client(Generate, gate);
        var idle = Client(Generate, gate);
        var iterator = active.GetStreamingResponseAsync([new(ChatRole.User, "pause")]).GetAsyncEnumerator();
        Check(await iterator.MoveNextAsync(), "active stream reaches a paused yield");
        await Task.Run(idle.Dispose).WaitAsync(TimeSpan.FromSeconds(2));
        Check(true, "idle conversation disposes without waiting for another conversation's model gate");
        await Task.Run(active.Dispose).WaitAsync(TimeSpan.FromSeconds(2));
        Check(!stopped && !sessions[0].IsDisposed, "active disposal cancels without synchronously waiting on its paused iterator");
        await iterator.DisposeAsync();
        Check(stopped && sessions[0].IsDisposed, "iterator cleanup completes deferred session disposal");
        using var recovery = Client(Generate, gate);
        Check((await recovery.GetResponseAsync([new(ChatRole.User, "recover")])).Text == "4", "another conversation reuses model gate after active disposal cleanup");
    }
}
