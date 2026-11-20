// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the root directory.

using System.Text.Json;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// An Ollama client (/api/chat/ollama) sends its history back the way the server returned it: an assistant
/// tool round with its <c>tool_calls</c> (arguments as a JSON object) and <c>thinking</c>, then the tool's
/// result. The parser used to read only role, content and images, so that round came back as an assistant
/// message without its call: the transcript store matched no recorded turn, and every request after a tool
/// call lost its conversation's cache scope and re-prefilled everything after the shared prefix (measured on
/// Qwen3.5-0.8B / ggml_metal: 265 of 967-1453 prompt tokens reused on each of the next seven requests).
/// </summary>
public class OllamaToolHistoryReuseTests
{
    private const string History = """
        [
          {"role": "user", "content": "What is the weather in Paris?"},
          {"role": "assistant", "content": "", "thinking": "The user wants the weather.",
           "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
          {"role": "tool", "content": "{\"weather\": \"18 C\"}", "tool_name": "get_weather"}
        ]
        """;

    [Fact]
    public void AnAssistantToolRound_KeepsItsCallsAndReasoning()
    {
        List<ChatMessage> messages = Parse(History);

        ChatMessage assistant = messages[1];
        ToolCall call = Assert.Single(assistant.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("{\"city\":\"Paris\"}", ConversationTranscriptStore.CanonicalArguments(call.Arguments));
        Assert.Equal("The user wants the weather.", assistant.Thinking);
        Assert.Equal("tool", messages[2].Role);
    }

    [Fact]
    public void TheRequestAfterAToolCall_ContinuesTheRecordedTurnAndItsScope()
    {
        var store = new ConversationTranscriptStore(16, 100_000);
        List<ChatMessage> first = Parse("""[{"role": "user", "content": "What is the weather in Paris?"}]""");
        var raw = new List<int> { 1, 2, 3, 4 };
        var call = new ToolCall { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } };
        store.Record(first,
            new ChatMessage { Role = "assistant", Content = "", RawOutputTokens = raw },
            new EmittedAssistantTurn("", new[] { call }, "The user wants the weather.", null, Cancelled: false),
            "conversation-scope");

        TranscriptAugmentation augmented = store.Augment(Parse(History));

        Assert.Equal(1, augmented.SplicedTurns);
        Assert.Equal("conversation-scope", augmented.InheritedScope);
        Assert.Same(raw, augmented.History[1].RawOutputTokens);
    }

    [Fact]
    public void AMalformedHistoryToolCall_IsARequestError()
        => Assert.Throws<JsonException>(() => Parse(
            """[{"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": {}}}]}]"""));

    private static List<ChatMessage> Parse(string json)
    {
        using var document = JsonDocument.Parse(json);
        // No images: the parser touches no upload storage.
        return ChatMessageParser.ParseOllama(document.RootElement, null);
    }
}
