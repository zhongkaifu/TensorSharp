// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Issue #386: a .NET client on the official OpenAI SDK, Microsoft.Extensions.AI and
// Microsoft.Agents.AI asks for reasoning (ChatOptions.Reasoning.Effort) and expects the
// model's thinking as TextReasoningContent. This program runs the reporter's code unchanged
// ("issue") and the same request over every client path ("matrix"): ChatClient and
// ResponsesClient, each as an agent (AsAIAgent) and as a bare IChatClient (AsIChatClient),
// streamed and not. It talks to one server only, the one --endpoint names.
//
//   dotnet run -- --endpoint http://127.0.0.1:5000/v1 --model Qwen3.8-9B-Q6_K [--scenario issue|matrix|NAME]
//                 [--effort none|low|medium|high|xhigh] [--max-tokens N] [--prompt TEXT] [--wire-log DIR]
//
// Exit code 0 when every scenario run got what its effort asks for: reasoning for any
// level but none, and no reasoning for none (each needs answer text too).
using System.ClientModel;
using System.ClientModel.Primitives;
using System.Diagnostics;
using System.Text;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Responses;

namespace OpenAISdkReasoningRepro;

public static class Program
{
    private const string IssuePrompt = "c#写一个计算当前月天数的方法";

    public static async Task<int> Main(string[] args)
    {
        var a = Args.Parse(args);
        Console.OutputEncoding = Encoding.UTF8;
        Console.WriteLine($"endpoint={a.Endpoint} model={a.Model} effort={a.Effort} maxTokens={a.MaxTokens?.ToString() ?? "(none)"}");
        Console.WriteLine($"packages: OpenAI {Version(typeof(ChatClient))}, Microsoft.Extensions.AI.OpenAI {Version(typeof(OpenAIClientExtensions))}, " +
                          $"Microsoft.Agents.AI {Version(typeof(ChatClientAgent))}");

        var options = new OpenAIClientOptions { Endpoint = new Uri(a.Endpoint), NetworkTimeout = TimeSpan.FromMinutes(20) };
        if (a.WireLog != null)
            options.Transport = new HttpClientPipelineTransport(new HttpClient(new WireLog(a.WireLog)) { Timeout = TimeSpan.FromMinutes(20) });

        var results = new List<Result>();
        foreach (string scenario in a.Scenarios)
        {
            Console.WriteLine();
            Console.WriteLine($"===== {scenario} =====");
            var watch = Stopwatch.StartNew();
            Result result;
            try
            {
                result = scenario switch
                {
                    "issue" => await IssueCode.RunAsync(options, a.Model, a.Prompt ?? IssuePrompt, a.EffortValue, thinkSwitch: false),
                    // The reporter's code with the thinking switch sent explicitly: what works on a
                    // server that does not read reasoning_effort as a request to think.
                    "issue-think" => await IssueCode.RunAsync(options, a.Model, a.Prompt ?? IssuePrompt, a.EffortValue, thinkSwitch: true),
                    _ => await RunScenarioAsync(scenario, options, a),
                };
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  error: {ex.GetType().Name}: {ex.Message}");
                result = new Result(scenario) { Error = ex.Message };
            }
            result.Seconds = watch.Elapsed.TotalSeconds;
            results.Add(result);
            Console.WriteLine(result.Describe(a.EffortValue));
        }

        Console.WriteLine();
        Console.WriteLine("===== summary =====");
        bool allPass = true;
        foreach (Result r in results)
        {
            bool pass = r.Passes(a.EffortValue);
            allPass &= pass;
            Console.WriteLine($"{(pass ? "PASS" : "FAIL")}  {r.Scenario,-26} reasoningParts={r.ReasoningParts,-5} reasoningChars={r.ReasoningChars,-6} " +
                              $"textParts={r.TextParts,-5} textChars={r.TextChars,-6} {r.Seconds,6:F1}s{(r.Error != null ? "  error=" + r.Error : "")}");
        }
        return allPass ? 0 : 1;
    }

    private static string Version(Type t) => t.Assembly.GetName().Version?.ToString() ?? "?";

    public static readonly string[] Matrix =
    {
        "chat-agent-stream", "chat-agent", "chat-ichat-stream", "chat-ichat",
        "responses-agent-stream", "responses-agent", "responses-ichat-stream", "responses-ichat",
    };

    /// <summary>Two streamed turns of one agent session: the second turn carries the first's
    /// history (or, on a Responses client that keeps stored output on, a previous_response_id).</summary>
    public static readonly string[] MultiTurn = { "chat-agent-2turns", "responses-agent-2turns", "responses-nostore-agent-2turns" };

    private static async Task<Result> RunScenarioAsync(string scenario, OpenAIClientOptions options, Args a)
    {
        var credential = new ApiKeyCredential("not-needed");
        bool responses = scenario.StartsWith("responses-", StringComparison.Ordinal);
        bool agent = scenario.Contains("-agent", StringComparison.Ordinal);
        bool stream = scenario.EndsWith("-stream", StringComparison.Ordinal);

        var chatOptions = new ChatOptions
        {
            Instructions = "You are a helpful assistant.",
            Reasoning = new ReasoningOptions { Effort = a.EffortValue },
            MaxOutputTokens = a.MaxTokens,
        };
        string prompt = a.Prompt ?? "What is 17 * 23? Answer in one sentence.";
        var result = new Result(scenario);

        if (scenario.EndsWith("-2turns", StringComparison.Ordinal))
        {
            var agentOptions = new ChatClientAgentOptions { Name = "LeadAgent", ChatOptions = chatOptions };
            AIAgent aiAgent = scenario switch
            {
                "chat-agent-2turns" => new ChatClient(a.Model, credential, options).AsAIAgent(agentOptions),
                "responses-agent-2turns" => new OpenAIClient(credential, options).GetResponsesClient().AsAIAgent(agentOptions, a.Model),
                // Stored output off: each turn sends the history instead of a previous_response_id.
                _ => new ChatClientAgent(new OpenAIClient(credential, options).GetResponsesClient()
                    .AsIChatClientWithStoredOutputDisabled(a.Model), agentOptions),
            };
            AgentSession session = await aiAgent.CreateSessionAsync();
            foreach (string turn in new[] { prompt, "Now add one to that number. Answer in one sentence." })
            {
                Console.WriteLine($"  turn: {turn}");
                await foreach (AgentResponseUpdate update in aiAgent.RunStreamingAsync(turn, session))
                    result.Add(update.Contents);
                result.EndTurn();
            }
            return result;
        }

        if (agent)
        {
            var agentOptions = new ChatClientAgentOptions { Name = "LeadAgent", ChatOptions = chatOptions };
            AIAgent aiAgent = responses
                ? new OpenAIClient(credential, options).GetResponsesClient().AsAIAgent(agentOptions, a.Model)
                : new ChatClient(a.Model, credential, options).AsAIAgent(agentOptions);
            if (stream)
            {
                await foreach (AgentResponseUpdate update in aiAgent.RunStreamingAsync(prompt))
                    result.Add(update.Contents);
            }
            else
            {
                AgentResponse response = await aiAgent.RunAsync(prompt);
                foreach (Microsoft.Extensions.AI.ChatMessage message in response.Messages)
                    result.Add(message.Contents);
            }
        }
        else
        {
            IChatClient client = responses
                ? new OpenAIClient(credential, options).GetResponsesClient().AsIChatClient(a.Model)
                : new ChatClient(a.Model, credential, options).AsIChatClient();
            var messages = new List<Microsoft.Extensions.AI.ChatMessage> { new(ChatRole.User, prompt) };
            if (stream)
            {
                await foreach (ChatResponseUpdate update in client.GetStreamingResponseAsync(messages, chatOptions))
                    result.Add(update.Contents);
            }
            else
            {
                ChatResponse response = await client.GetResponseAsync(messages, chatOptions);
                foreach (Microsoft.Extensions.AI.ChatMessage message in response.Messages)
                    result.Add(message.Contents);
            }
        }
        return result;
    }

    private sealed record Args(string Endpoint, string Model, string[] Scenarios, string Effort, int? MaxTokens,
        string? Prompt, string? WireLog)
    {
        public ReasoningEffort EffortValue => Effort switch
        {
            "none" => ReasoningEffort.None,
            "low" => ReasoningEffort.Low,
            "medium" => ReasoningEffort.Medium,
            "high" => ReasoningEffort.High,
            "xhigh" => ReasoningEffort.ExtraHigh,
            _ => throw new ArgumentException($"--effort {Effort}: use none, low, medium, high or xhigh"),
        };

        public static Args Parse(string[] args)
        {
            string endpoint = "http://127.0.0.1:5000/v1", model = "Qwen3.8-9B-Q6_K", scenario = "all", effort = "high";
            int? maxTokens = null;
            string? prompt = null, wireLog = null;
            for (int i = 0; i < args.Length; i++)
            {
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
                switch (args[i])
                {
                    case "--endpoint": endpoint = Next(); break;
                    case "--model": model = Next(); break;
                    case "--scenario": scenario = Next(); break;
                    case "--effort": effort = Next(); break;
                    case "--max-tokens": maxTokens = int.Parse(Next()); break;
                    case "--prompt": prompt = Next(); break;
                    case "--wire-log": wireLog = Next(); break;
                    default: throw new ArgumentException($"unknown argument {args[i]}");
                }
            }
            string[] scenarios = scenario switch
            {
                "all" => new[] { "issue" }.Concat(Matrix).Concat(MultiTurn).ToArray(),
                "matrix" => Matrix,
                "multiturn" => MultiTurn,
                _ => scenario.Split(','),
            };
            foreach (string s in scenarios)
                if (s != "issue" && s != "issue-think" && !Matrix.Contains(s) && !MultiTurn.Contains(s))
                    throw new ArgumentException($"unknown scenario {s}: issue, issue-think, matrix, multiturn, all or one of " +
                                                string.Join(", ", Matrix.Concat(MultiTurn)));
            var parsed = new Args(endpoint, model, scenarios, effort, maxTokens, prompt, wireLog);
            _ = parsed.EffortValue;
            return parsed;
        }
    }
}

/// <summary>What one scenario received, by content type.</summary>
public sealed class Result(string scenario)
{
    public string Scenario { get; } = scenario;
    public int ReasoningParts { get; private set; }
    public int TextParts { get; private set; }
    public string? Error { get; set; }
    public double Seconds { get; set; }
    private readonly StringBuilder _reasoning = new();
    private readonly StringBuilder _text = new();
    public int ReasoningChars => _reasoning.Length;
    public int TextChars => _text.Length;
    private readonly List<(int Reasoning, int Text)> _turns = new();
    private int _turnReasoningStart, _turnTextStart;

    /// <summary>Close a turn of a multi-turn scenario; each turn must pass on its own.</summary>
    public void EndTurn()
    {
        _turns.Add((_reasoning.Length - _turnReasoningStart, _text.Length - _turnTextStart));
        _turnReasoningStart = _reasoning.Length;
        _turnTextStart = _text.Length;
    }

    public void Add(IEnumerable<AIContent> contents)
    {
        foreach (AIContent content in contents)
            Add(content);
    }

    public void Add(AIContent content)
    {
        // TextReasoningContent is not a TextContent, so the order does not matter.
        switch (content)
        {
            case TextReasoningContent reasoning when !string.IsNullOrEmpty(reasoning.Text):
                ReasoningParts++;
                _reasoning.Append(reasoning.Text);
                break;
            case TextContent text when !string.IsNullOrEmpty(text.Text):
                TextParts++;
                _text.Append(text.Text);
                break;
        }
    }

    public bool Passes(ReasoningEffort effort)
    {
        bool Ok(int reasoning, int text) => text > 0 && (effort == ReasoningEffort.None ? reasoning == 0 : reasoning > 0);
        return Error == null && (_turns.Count == 0 ? Ok(ReasoningChars, TextChars) : _turns.All(t => Ok(t.Reasoning, t.Text)));
    }

    public string Describe(ReasoningEffort effort)
        => $"  TextReasoningContent: {ReasoningParts} parts, {ReasoningChars} chars: {Clip(_reasoning.ToString())}\n"
         + $"  TextContent:          {TextParts} parts, {TextChars} chars: {Clip(_text.ToString())}\n"
         + (_turns.Count > 0 ? $"  turns (reasoning chars, text chars): {string.Join(" ", _turns.Select(t => $"({t.Reasoning}, {t.Text})"))}\n" : "")
         + $"  => {(Passes(effort) ? "PASS" : "FAIL")} (effort {effort}: expects {(effort == ReasoningEffort.None ? "no reasoning" : "reasoning")} and answer text)";

    private static string Clip(string s)
    {
        s = s.Replace("\r", "").Replace("\n", "\\n");
        return s.Length <= 160 ? $"\"{s}\"" : $"\"{s[..120]} ... {s[^30..]}\"";
    }
}

/// <summary>
/// The reporter's program from issue #386, unchanged apart from the endpoint and model
/// coming from the command line and a <see cref="Result"/> recording what the handler saw.
/// </summary>
public static class IssueCode
{
    public static class TerminalColor
    {
        public static void Reset() => Console.ResetColor();
        public static void Reasoning() => Console.ForegroundColor = ConsoleColor.Blue;
        public static void FunctionCall() => Console.ForegroundColor = ConsoleColor.DarkYellow;
        public static void FunctionResult() => Console.ForegroundColor = ConsoleColor.Green;
    }

    public class TokenPerformanceTracker
    {
        private Stopwatch? _stopwatch;
        private double _firstTokenLatency;
        private readonly List<double> _tokenIntervals = new();
        public int TotalTokens { get; private set; }

        public void Start()
        {
            _stopwatch = Stopwatch.StartNew();
            _firstTokenLatency = 0;
            _tokenIntervals.Clear();
            TotalTokens = 0;
        }

        public void RecordToken()
        {
            if (_stopwatch == null) return;
            if (TotalTokens == 0)
                _firstTokenLatency = _stopwatch.Elapsed.TotalSeconds;
            else
                _tokenIntervals.Add(_stopwatch.Elapsed.TotalSeconds);
            _stopwatch.Restart();
            TotalTokens++;
        }

        public void PrintStats()
        {
            Console.WriteLine($"\n{new string('=', 25)} Ollama 流式性能统计 (Chunk) {new string('=', 25)}");
            Console.WriteLine($"首 Chunk 延迟: {_firstTokenLatency:F3}s");
            double avgIntervalMs = 0;
            double throughput = 0;
            if (_tokenIntervals.Count > 0)
            {
                avgIntervalMs = _tokenIntervals.Average() * 1000;
                if (avgIntervalMs > 0)
                    throughput = 1000.0 / avgIntervalMs;
            }
            Console.WriteLine($"平均 Chunk 间隔: {avgIntervalMs:F2}ms");
            Console.WriteLine($"平均生成速度: {throughput:F2} chunks/s");
            Console.WriteLine($"总输出 Chunks: {TotalTokens}");
            Console.WriteLine($"{new string('=', 80)}");
        }
    }

    public class StreamOutputHandler(Result seen)
    {
        private bool _lastIsReasoning;
        private bool _needNewLine;

        public void HandleContent(AIContent content)
        {
            seen.Add(content);
            if (_lastIsReasoning && content is not TextReasoningContent)
            {
                _needNewLine = true;
                _lastIsReasoning = false;
            }
            if (_needNewLine)
            {
                Console.WriteLine();
                _needNewLine = false;
            }
            switch (content)
            {
                case TextReasoningContent reasoning:
                    HandleReasoning(reasoning.Text);
                    break;
                case TextContent text:
                    HandleText(text.Text);
                    break;
                case FunctionCallContent:
                    HandleFunctionCall();
                    break;
                case FunctionResultContent:
                    HandleFunctionResult();
                    break;
            }
        }

        private void HandleReasoning(string text)
        {
            TerminalColor.Reasoning();
            Console.Write(text);
            TerminalColor.Reset();
            Console.Out.Flush();
            _lastIsReasoning = true;
        }

        private static void HandleText(string text)
        {
            Console.Write(text);
            Console.Out.Flush();
        }

        private static void HandleFunctionCall()
        {
            TerminalColor.FunctionCall();
            Console.Write("[Function Call]");
            TerminalColor.Reset();
            Console.Out.Flush();
        }

        private static void HandleFunctionResult()
        {
            TerminalColor.FunctionResult();
            Console.Write("[Function Result]");
            TerminalColor.Reset();
            Console.Out.Flush();
        }
    }

    public static async Task<Result> RunAsync(OpenAIClientOptions options, string model, string prompt, ReasoningEffort effort,
        bool thinkSwitch)
    {
        var client = new ChatClient(
            model,
            new ApiKeyCredential("not-needed"),
            options);
        var chatOptions = new ChatOptions
        {
            Instructions = "你是一个助手",
            Reasoning = new() { Effort = effort },
        };
        // MEAI does not copy ChatOptions.AdditionalProperties into the request; a field of
        // the server's own goes in through the request object's JSON patch.
        if (thinkSwitch)
            chatOptions.RawRepresentationFactory = _ =>
            {
                var request = new ChatCompletionOptions();
                request.Patch.Set("$.think"u8, true);
                return request;
            };
        var mainAgent = client.AsAIAgent(new ChatClientAgentOptions
        {
            Name = "LeadAgent",
            ChatOptions = chatOptions,
        });

        var seen = new Result(thinkSwitch ? "issue-think" : "issue");
        var perfTracker = new TokenPerformanceTracker();
        var outputHandler = new StreamOutputHandler(seen);
        await foreach (var chunk in mainAgent.RunStreamingAsync(prompt))
        {
            foreach (var content in chunk.Contents)
            {
                outputHandler.HandleContent(content);
                if (content is TextContent)
                    perfTracker.RecordToken();
            }
        }
        perfTracker.PrintStats();
        return seen;
    }
}

/// <summary>Saves each request body and the response body as it arrives (streams included),
/// numbered, so a run's wire traffic can be read next to its result.</summary>
internal sealed class WireLog : DelegatingHandler
{
    private readonly string _dir;
    private int _n;

    public WireLog(string dir) : base(new HttpClientHandler())
    {
        _dir = dir;
        Directory.CreateDirectory(dir);
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        int n = Interlocked.Increment(ref _n);
        string stem = Path.Combine(_dir, $"{n:D2}-{request.RequestUri?.AbsolutePath.Trim('/').Replace('/', '_')}");
        string body = request.Content != null ? await request.Content.ReadAsStringAsync(cancellationToken) : "";
        await File.WriteAllTextAsync(stem + ".request.json", body, cancellationToken);
        HttpResponseMessage response = await base.SendAsync(request, cancellationToken);
        Stream inner = await response.Content.ReadAsStreamAsync(cancellationToken);
        var tee = new StreamContent(new TeeStream(inner, File.Create(stem + $".response-{(int)response.StatusCode}.txt")));
        foreach (var header in response.Content.Headers)
            tee.Headers.TryAddWithoutValidation(header.Key, header.Value);
        response.Content = tee;
        return response;
    }

    private sealed class TeeStream(Stream inner, Stream copy) : Stream
    {
        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();

        public override int Read(byte[] buffer, int offset, int count)
        {
            int read = inner.Read(buffer, offset, count);
            copy.Write(buffer, offset, read);
            copy.Flush();
            return read;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            int read = await inner.ReadAsync(buffer, cancellationToken);
            await copy.WriteAsync(buffer[..read], cancellationToken);
            await copy.FlushAsync(cancellationToken);
            return read;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                inner.Dispose();
                copy.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}
