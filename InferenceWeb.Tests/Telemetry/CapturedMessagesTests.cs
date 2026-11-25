// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// <c>gen_ai.input.messages</c> and <c>gen_ai.output.messages</c> follow the OpenTelemetry GenAI
/// message schemas: every message has <c>role</c> and <c>parts</c>, text is a <c>text</c> part,
/// tool calls and results are <c>tool_call</c> and <c>tool_call_response</c> parts. GenAI views
/// such as the Application Insights agent view show nothing for messages without <c>parts</c>.
/// </summary>
public class CapturedMessagesTests
{
    [Fact]
    public void InputMessagesCarryTheirTextAsTextParts()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Describe the clip." },
            new() { Role = "user", Content = "What happens?" },
        };

        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(messages, null));

        Assert.Equal(2, root.GetArrayLength());
        AssertTextMessage(root[0], "system", "Describe the clip.");
        AssertTextMessage(root[1], "user", "What happens?");
        Assert.All(root.EnumerateArray(), m => Assert.False(m.TryGetProperty("content", out _)));
    }

    [Fact]
    public void TheAnswerIsOneAssistantMessageWithATextPart()
    {
        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(null, "A person walks a dog."));

        AssertTextMessage(Assert.Single(root.EnumerateArray()), "assistant", "A person walks a dog.");
    }

    [Fact]
    public void ToolCallsAndToolResultsBecomeToolParts()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Weather in Paris?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls = new List<ToolCall>
                {
                    new() { Id = "call_1", Name = "get_weather", Arguments = new Dictionary<string, object?> { ["location"] = "Paris" } },
                },
            },
            new() { Role = "tool", ToolCallId = "call_1", Content = "rainy, 57°F" },
        };

        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(messages, null));

        // The assistant turn only called a tool: no empty text part next to the call.
        JsonElement call = Assert.Single(root[1].GetProperty("parts").EnumerateArray());
        Assert.Equal("tool_call", call.GetProperty("type").GetString());
        Assert.Equal("call_1", call.GetProperty("id").GetString());
        Assert.Equal("get_weather", call.GetProperty("name").GetString());
        Assert.Equal("Paris", call.GetProperty("arguments").GetProperty("location").GetString());

        Assert.Equal("tool", root[2].GetProperty("role").GetString());
        JsonElement result = Assert.Single(root[2].GetProperty("parts").EnumerateArray());
        Assert.Equal("tool_call_response", result.GetProperty("type").GetString());
        Assert.Equal("call_1", result.GetProperty("id").GetString());
        Assert.Equal("rainy, 57°F", result.GetProperty("response").GetString());
    }

    [Fact]
    public void AnEmptyMessageKeepsAnEmptyTextPart()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = null } };

        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(messages, null));

        AssertTextMessage(Assert.Single(root.EnumerateArray()), "user", "");
    }

    [Fact]
    public void ALongHistoryKeepsItsNewestMessagesAndTheEndOfTheOldestKept()
    {
        string old = new string('a', 40 * 1024);
        string kept = new string('b', 40 * 1024) + "END";
        string newest = new string('c', 40 * 1024);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = old },
            new() { Role = "assistant", Content = kept },
            new() { Role = "user", Content = newest },
        };

        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(messages, null));

        Assert.Equal(2, root.GetArrayLength());
        string first = TextOf(root[0]);
        Assert.StartsWith("…", first);
        Assert.EndsWith("END", first);
        Assert.Equal(newest, TextOf(root[1]));
        Assert.Equal(64 * 1024, root.EnumerateArray().Sum(m => TextOf(m).TrimStart('…').Length));
    }

    [Fact]
    public void ALongAnswerKeepsItsBeginning()
    {
        string answer = "START" + new string('x', 70 * 1024);

        JsonElement root = Parse(ChatGenerationPipeline.CapturedMessagesJson(null, answer));

        string text = TextOf(Assert.Single(root.EnumerateArray()));
        Assert.StartsWith("START", text);
        Assert.EndsWith("…", text);
        Assert.Equal(64 * 1024 + 1, text.Length);
    }

    private static JsonElement Parse(string? json)
    {
        Assert.NotNull(json);
        JsonElement root = JsonDocument.Parse(json!).RootElement;
        Assert.Equal(JsonValueKind.Array, root.ValueKind);
        return root;
    }

    private static void AssertTextMessage(JsonElement message, string role, string text)
    {
        Assert.Equal(role, message.GetProperty("role").GetString());
        Assert.Equal(text, TextOf(message));
    }

    private static string TextOf(JsonElement message)
    {
        JsonElement part = Assert.Single(message.GetProperty("parts").EnumerateArray());
        Assert.Equal("text", part.GetProperty("type").GetString());
        return part.GetProperty("content").GetString()!;
    }
}
