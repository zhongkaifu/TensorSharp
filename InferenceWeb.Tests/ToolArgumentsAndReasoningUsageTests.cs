// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text;
using System.Text.Json;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

/// <summary>
/// OpenAI-compatible responses carry a tool call's <c>arguments</c> as the raw UTF-8 JSON OpenAI and llama.cpp write (it was
/// System.Text.Json's default escaping: <c>北京</c>, <c>'</c>, <c>+</c>), and the Responses API's
/// <c>reasoning_tokens</c> counts the reasoning a reply generated (it was always 0).
/// </summary>
public sealed class ToolArgumentsAndReasoningUsageTests
{
    private static ToolCall Call()
    {
        using JsonDocument doc = JsonDocument.Parse("""{"note":"it's 1+1 <b> & \"quoted\" \\ back"}""");
        return new ToolCall
        {
            Name = "get_weather",
            Index = 0,
            Arguments = new Dictionary<string, object?>
            {
                ["city"] = "北京",
                ["emoji"] = "天气 ☀️",
                ["query"] = "it's 1+1 <b> & \"quoted\" \\ back\n",
                ["days"] = 3,
                ["nested"] = doc.RootElement.Clone(),
            },
        };
    }

    private const string RawArguments =
        "{\"city\":\"北京\",\"emoji\":\"天气 ☀️\",\"query\":\"it's 1+1 <b> & \\\"quoted\\\" \\\\ back\\n\",\"days\":3,"
        + "\"nested\":{\"note\":\"it's 1+1 <b> & \\\"quoted\\\" \\\\ back\"}}";

    [Fact]
    public void ArgumentsAreRawUtf8Json()
    {
        string json = ToolCallArgumentsJson.Serialize(Call().Arguments);
        Assert.Equal(RawArguments, json);
        // Still JSON, with the same values.
        using JsonDocument doc = JsonDocument.Parse(json);
        Assert.Equal("北京", doc.RootElement.GetProperty("city").GetString());
        Assert.Equal("it's 1+1 <b> & \"quoted\" \\ back\n", doc.RootElement.GetProperty("query").GetString());
    }

    /// <summary>Every OpenAI-compatible place a call goes out: the chat message, the stream's delta, the Responses item.</summary>
    [Fact]
    public void EveryOpenAIShapeCarriesTheRawArguments()
    {
        var calls = new List<ToolCall> { Call() };
        string Arguments(object shape, Func<JsonElement, JsonElement> pick)
        {
            using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(shape));
            return pick(doc.RootElement).GetString()!;
        }
        Assert.Equal(RawArguments, Arguments(OpenAIResponseFactory.ParsedAssistantMessage("", null, calls),
            m => m.GetProperty("tool_calls")[0].GetProperty("function").GetProperty("arguments")));
        Assert.Equal(RawArguments, Arguments(OpenAIResponseFactory.ToolCallsChunk("r", "m", calls),
            c => c.GetProperty("choices")[0].GetProperty("delta").GetProperty("tool_calls")[0].GetProperty("function").GetProperty("arguments")));
        Assert.Equal(RawArguments, Arguments(OpenAIResponsesFactory.FunctionCallItem("fc", "call", calls[0]),
            i => i.GetProperty("arguments")));
    }

    // ---------------------------------------------------------------- reasoning_tokens

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public bool IsEos(int id) => id == 0;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => string.Concat(tokens.Select(t => (char)t));
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }

    private static int Count(string arch, string reply, bool thinking, string? suffix)
        => ReasoningTokenCount.Count(arch, new CharTokenizer(), reply.Select(c => (int)c).ToList(), thinking, null, suffix);

    [Theory]
    // Thinking on: the reasoning and its close (one character a token here), not the answer.
    [InlineData("qwen35", "abc</think>xyz", true, null, 11)]
    [InlineData("lfm3", "abc</think>xyz", true, "<think>", 11)]
    // LFM3 thinking off that reasoned past the closed block: the stretch a close showed was reasoning.
    [InlineData("lfm3", "Brazil\nArgentina\nChile</think>Brazil\nArgentina\nChile", false, "<think></think>", 30)]
    // An answer, and a reply that opened a block of its own.
    [InlineData("lfm3", "The capital is Canberra.", false, "<think></think>", 0)]
    [InlineData("lfm3", "<think>Au is gold.</think>Au", false, "<think></think>", 26)]
    // Reasoning that never closed: every token was reasoning.
    [InlineData("qwen35", "still thinking", true, null, 14)]
    public void ReasoningTokensAreTheReasoningsTokens(string arch, string reply, bool thinking, string? suffix, int expected)
        => Assert.Equal(expected, Count(arch, reply, thinking, suffix));

    [Fact]
    public void TheResponsesUsageReportsThem()
    {
        object response = OpenAIResponsesFactory.Response("resp", "m", "completed", null, 100, Array.Empty<object>(), false,
            SamplingConfig.Greedy, promptTokens: 10, evalTokens: 40, kvCacheReusedTokens: 0, reasoningTokens: 30);
        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(response));
        JsonElement usage = doc.RootElement.GetProperty("usage");
        Assert.Equal(30, usage.GetProperty("output_tokens_details").GetProperty("reasoning_tokens").GetInt32());
        Assert.Equal(40, usage.GetProperty("output_tokens").GetInt32());
    }
}
