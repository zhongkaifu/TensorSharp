// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// End-to-end gpt-oss / Harmony function-calling test against a real GGUF.
// Validates that (1) the tool-enabled prompt makes the model emit a tool call,
// (2) generation stops on <|call|> (the stop-token fix), (3) the parser recovers
// the call, and (4) feeding the tool result back yields a grounded final answer.
// Also reports prefill/decode throughput.
//
// Opt-in via TS_TEST_MODEL_DIR pointing at a directory containing gpt-oss-*.gguf.
// Slow (loads a multi-GB model).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class HarmonyToolCallIntegrationTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public HarmonyToolCallIntegrationTests(ITestOutputHelper output) { _output = output; }

    private static List<ToolFunction> WeatherTool() => new()
    {
        new ToolFunction
        {
            Name = "get_current_weather",
            Description = "Get the current weather for a given location",
            Parameters = new Dictionary<string, ToolParameter>
            {
                ["location"] = new ToolParameter
                {
                    Type = "string",
                    Description = "The city and state, e.g. San Francisco, CA",
                },
                ["unit"] = new ToolParameter
                {
                    Type = "string",
                    Description = "Temperature unit",
                    Enum = new List<string> { "celsius", "fahrenheit" },
                },
            },
            Required = new List<string> { "location" },
        },
    };

    [Fact]
    public async Task GptOss_ToolCall_RoundTrip()
    {
        var modelPath = FindGptOss();
        if (modelPath == null) { _output.WriteLine("[harmony-tool] no model; skipping"); return; }
        _output.WriteLine($"[harmony-tool] loading {Path.GetFileName(modelPath)}");

        using var ctx = new Ctx(modelPath);
        var tools = WeatherTool();

        _output.WriteLine($"[harmony-tool] architecture={ctx.Model.Config?.Architecture} " +
                          $"eosTokens=[{string.Join(",", EosIds(ctx))}]");

        // -- Turn 1: user asks; expect a tool call. ------------------------
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the weather like in San Francisco, CA right now? Use the tool." },
        };

        var (text1, outTokens1, prefill1, prefillTps, decodeTps) =
            await Generate(ctx, history, tools, maxNewTokens: 512);

        _output.WriteLine($"[harmony-tool] turn1 prompt={prefill1} tok, out={outTokens1} tok, " +
                          $"prefill={prefillTps:F1} tok/s, decode={decodeTps:F1} tok/s");
        _output.WriteLine($"[harmony-tool] turn1 raw decode:\n{Trim(text1, 600)}");

        var parser1 = new HarmonyOutputParser();
        parser1.Init(enableThinking: true, tools: tools);
        var parsed1 = parser1.Add(text1, done: true);

        _output.WriteLine($"[harmony-tool] turn1 thinking: \"{Trim(parsed1.Thinking, 200)}\"");
        Xunit.Assert.NotNull(parsed1.ToolCalls);
        var call = parsed1.ToolCalls![0];
        _output.WriteLine($"[harmony-tool] turn1 tool call: {call}");

        Xunit.Assert.Equal("get_current_weather", call.Name);
        Xunit.Assert.True(call.Arguments.ContainsKey("location"),
            "tool call must include a 'location' argument");
        string loc = call.Arguments["location"]?.ToString() ?? "";
        Xunit.Assert.Contains("San Francisco", loc, StringComparison.OrdinalIgnoreCase);

        // Generation should have stopped on <|call|> rather than running to the cap.
        Xunit.Assert.True(outTokens1 < 512,
            "generation should stop at <|call|>, not hit the token cap");

        // -- Turn 2: feed the tool result; expect a grounded final answer. -
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Thinking = parsed1.Thinking,
            ToolCalls = parsed1.ToolCalls,
        });
        history.Add(new ChatMessage
        {
            Role = "tool",
            Content = "{\"temperature\": 12, \"unit\": \"celsius\", \"conditions\": \"foggy\"}",
        });

        var (text2, outTokens2, prefill2, prefillTps2, decodeTps2) =
            await Generate(ctx, history, tools, maxNewTokens: 512);

        var parser2 = new HarmonyOutputParser();
        parser2.Init(enableThinking: true, tools: tools);
        var parsed2 = parser2.Add(text2, done: true);

        _output.WriteLine($"[harmony-tool] turn2 prompt={prefill2} tok, out={outTokens2} tok, " +
                          $"prefill={prefillTps2:F1} tok/s, decode={decodeTps2:F1} tok/s");
        _output.WriteLine($"[harmony-tool] turn2 final answer: \"{Trim(parsed2.Content, 400)}\"");

        Xunit.Assert.False(string.IsNullOrWhiteSpace(parsed2.Content),
            "model should produce a final answer after the tool result");
        // The grounded answer should reflect the tool output (12 / foggy).
        bool grounded = parsed2.Content.Contains("12") ||
                        parsed2.Content.Contains("fog", StringComparison.OrdinalIgnoreCase);
        Xunit.Assert.True(grounded,
            $"final answer should reflect the tool result; got: {Trim(parsed2.Content, 200)}");
    }

    /// <summary>Render → submit → collect, returning decoded text, counts, and throughput.</summary>
    private async Task<(string text, int outTokens, int prefillTokens, double prefillTps, double decodeTps)>
        Generate(Ctx ctx, List<ChatMessage> history, List<ToolFunction> tools, int maxNewTokens)
    {
        ctx.Model.ResetKVCache();

        var promptTokens = ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer, ctx.Model.Config?.ChatTemplate, history,
            ctx.Model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true, tools: tools, enableThinking: true);

        string reqId = $"harmony-tool-{Guid.NewGuid():N}";
        var seq = new SequenceState(reqId, promptTokens, maxNewTokens, ctx.BlockSize, SamplingConfig.Greedy);

        var sw = Stopwatch.StartNew();
        double firstTokenMs = -1;
        var handle = ctx.Engine.SubmitRequest(seq);
        var outs = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
            {
                if (firstTokenMs < 0) firstTokenMs = sw.Elapsed.TotalMilliseconds;
                outs.Add(tok);
            }
        }
        catch { }
        await handle.Completion;
        double totalMs = sw.Elapsed.TotalMilliseconds;

        double prefillTps = firstTokenMs > 0 ? promptTokens.Count / (firstTokenMs / 1000.0) : 0;
        double decodeMs = totalMs - (firstTokenMs > 0 ? firstTokenMs : 0);
        double decodeTps = (outs.Count > 1 && decodeMs > 0) ? (outs.Count - 1) / (decodeMs / 1000.0) : 0;

        string text = ctx.Model.Tokenizer.Decode(outs);
        return (text, outs.Count, promptTokens.Count, prefillTps, decodeTps);
    }

    private static IEnumerable<int> EosIds(Ctx ctx)
    {
        // Probe which token ids are treated as stop tokens around the Harmony range.
        for (int id = 199998; id <= 200018; id++)
            if (ctx.Model.Tokenizer.IsEos(id)) yield return id;
    }

    private static string Trim(string s, int len)
        => string.IsNullOrEmpty(s) ? string.Empty
           : (s.Length <= len ? s : s.Substring(0, len) + "...");

    private static string FindGptOss()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").Where(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return (n.Contains("gpt-oss") || n.Contains("gpt_oss") || n.Contains("gptoss"))
                && !n.Contains("mmproj");
        }).OrderBy(p => Path.GetFileName(p)).FirstOrDefault();
    }

    private sealed class Ctx : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }

        public Ctx(string modelPath)
        {
            BackendType backend = OperatingSystem.IsMacOS()
                ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 256,
                BlockSize = BlockSize,
                EnablePrefixCaching = false,
                DecodeQuantumTokens = BlockSize,
            };
            Engine = new InferenceEngine(Model, cfg, NullLogger.Instance);
        }

        public void Dispose()
        {
            Engine?.Dispose();
            Model?.Dispose();
        }
    }
}
