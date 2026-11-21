// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Inkling 2's chat protocol: the renderer against the published template (the expected prompts are what
/// <c>eng/inkling2/render_template_reference.py</c> renders with jinja2, as transformers does) and the reply parser.
/// </summary>
public sealed class Inkling2ChatProtocolTests
{
    // render_template_reference.py chat_template.jinja on the five conversations below, in order.
    private static readonly string[] Reference =
    {
        "<|message_system|><|content_text|>You are a terse assistant.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|><|message_user|><|content_text|>Name three prime numbers.<|end_message|>",
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hi<|end_message|>",
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>What is 2+2?<|end_message|><|message_model|><|content_thinking|>Simple sum.<|end_message|><|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>And 3+3?<|end_message|>",
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Current weather for a city.\",\"name\":\"get_weather\",\"parameters\":{\"properties\":{\"city\":{\"description\":\"City name\",\"type\":\"string\"},\"unit\":{\"enum\":[\"c\",\"f\"],\"type\":\"string\"}},\"required\":[\"city\"],\"type\":\"object\"},\"type\":\"function\"},{\"description\":\"\",\"name\":\"now\",\"parameters\":{\"properties\":{},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Use tools.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Weather in Paris and Zürich?<|end_message|><|message_model|><|content_thinking|>Two calls.<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"opts\":{\"z\":1,\"a\":[1.5,\"x<y\"]},\"unit\":\"c\"}}<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Zürich\"}}<|end_message|><|content_model_end_sampling|><|message_tool|>get_weather<|content_text|>18C<|end_message|><|message_tool|>get_weather<|content_text|>12C<|end_message|>",
        "<|message_system|>tool_declare<|content_xml|>[{\"description\":\"\",\"name\":\"ls\",\"parameters\":{\"properties\":{},\"type\":\"object\"},\"type\":\"function\"}]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>List files<|end_message|><|message_model|><|content_text|>Listing.<|end_message|><|message_model|>ls<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":{}}<|end_message|><|content_model_end_sampling|><|message_tool|>ls<|content_text|>x.txt<|end_message|><|message_model|><|content_text|>One file: x.txt<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>Thanks<|end_message|>",
    };

    private static readonly List<ToolFunction> WeatherTools = ToolFunction.ParseList("""
        [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["c", "f"]}}, "required": ["city"]}}},
         {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}]
        """);

    private static ChatMessage M(string role, string content, string thinking = null, List<ToolCall> calls = null, string callId = null)
        => new() { Role = role, Content = content, Thinking = thinking, ToolCalls = calls, ToolCallId = callId };

    [Fact]
    public void RendersAsThePublishedTemplate()
    {
        var weatherCalls = new List<ToolCall>
        {
            new()
            {
                Id = "call_1", Name = "get_weather",
                Arguments = new Dictionary<string, object>
                {
                    ["unit"] = "c", ["city"] = "Paris",
                    ["opts"] = new Dictionary<string, object> { ["z"] = 1L, ["a"] = new List<object> { 1.5, "x<y" } },
                },
            },
            new() { Id = "call_2", Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Zürich" } },
        };
        var lsTools = ToolFunction.ParseList("""[{"type": "function", "function": {"name": "ls", "description": "", "parameters": {"type": "object", "properties": {}}}}]""");
        var cases = new (List<ChatMessage> Messages, List<ToolFunction> Tools, string Effort)[]
        {
            (new() { M("system", "You are a terse assistant."), M("user", "Name three prime numbers.") }, null, "low"),
            (new() { M("user", "Hi") }, null, null),
            (new() { M("user", "What is 2+2?"), M("assistant", "4", thinking: "Simple sum."), M("user", "And 3+3?") }, null, "high"),
            (new()
            {
                M("system", "Use tools."), M("user", "Weather in Paris and Zürich?"),
                M("assistant", null, thinking: "Two calls.", calls: weatherCalls),
                M("tool", "18C", callId: "call_1"), M("tool", "12C", callId: "call_2"),
            }, WeatherTools, "medium"),
            (new()
            {
                M("user", "List files"),
                M("assistant", "Listing.", calls: new() { new() { Id = "a", Name = "ls", Arguments = new Dictionary<string, object>() } }),
                M("tool", "x.txt", callId: "a"), M("assistant", "One file: x.txt"), M("user", "Thanks"),
            }, lsTools, null),
        };
        Assert.Equal(Reference.Length, cases.Length);
        for (int i = 0; i < cases.Length; i++)
            Assert.Equal(Reference[i], ChatTemplate.RenderInkling2(cases[i].Messages, cases[i].Tools, cases[i].Effort));
    }

    [Fact]
    public void TheProtocolRendersThroughItsOwnRenderer()
    {
        var messages = new List<ChatMessage> { M("user", "Hi") };
        foreach (string arch in new[] { "inkling2", "inkling2_loop" })
            Assert.Equal(Reference[1], ChatTemplate.RenderFromGgufTemplate("{{ not used }}", messages, true, arch));
    }

    [Fact]
    public void ToolResultsWithoutIdsTakeTheirCallsInOrder()
    {
        var calls = new List<ToolCall> { new() { Name = "a" }, new() { Name = "b" } };
        string prompt = ChatTemplate.RenderInkling2(new() { M("user", "x"), M("assistant", null, calls: calls), M("tool", "1"), M("tool", "2") });
        Assert.EndsWith("<|message_tool|>a<|content_text|>1<|end_message|><|message_tool|>b<|content_text|>2<|end_message|>", prompt);
    }

    [Fact]
    public void ASplicedTurnIsItsRawTokensAndTheEndOfSampling()
    {
        // The generated tokens of a past turn already hold its framing; the template's would double it.
        string placeholder = KVCachePromptRenderer.MakePlaceholder(0);
        string prompt = ChatTemplate.RenderInkling2(new() { M("user", "x"), M("assistant", placeholder), M("user", "y") });
        Assert.Contains("<|content_text|>x<|end_message|>" + placeholder + "<|content_model_end_sampling|><|message_user|>", prompt);
    }

    private static IOutputParser NewParser()
    {
        IOutputParser parser = ChatProtocolRegistry.For("inkling2_loop").CreateOutputParser();
        parser.Init(enableThinking: true, tools: null);
        return parser;
    }

    private static (string Content, string Thinking, List<ToolCall> Calls) Parse(string reply, int piece)
    {
        IOutputParser parser = NewParser();
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var calls = new List<ToolCall>();
        for (int i = 0; i < reply.Length; i += piece)
        {
            ParsedOutput r = parser.Add(reply.Substring(i, System.Math.Min(piece, reply.Length - i)), done: i + piece >= reply.Length);
            content.Append(r.Content);
            thinking.Append(r.Thinking);
            if (r.ToolCalls != null) calls.AddRange(r.ToolCalls);
        }
        return (content.ToString(), thinking.ToString(), calls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(1000)]
    public void SplitsReasoningFromTheAnswer(int piece)
    {
        // As the server streamed it: the model writes both messages; the end-of-sampling token closes the reply.
        const string reply = "<|message_model|><|content_thinking|>17*23 = 391.<|end_message|><|message_model|><|content_text|>391<|end_message|><|content_model_end_sampling|>";
        var (content, thinking, calls) = Parse(reply, piece);
        Assert.Equal("391", content);
        Assert.Equal("17*23 = 391.", thinking);
        Assert.Empty(calls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(1000)]
    public void ReadsToolCalls(int piece)
    {
        const string reply = "<|message_model|><|content_thinking|>Two calls.<|end_message|>" +
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"opts\":{\"z\":1}}}<|end_message|>" +
            "<|message_model|>get_weather<|content_invoke_tool_json|>{\"name\":\"get_weather\",\"args\":{\"city\":\"Zürich\"}}<|end_message|>";
        var (content, thinking, calls) = Parse(reply, piece);
        Assert.Equal("", content);
        Assert.Equal("Two calls.", thinking);
        Assert.Equal(2, calls.Count);
        Assert.All(calls, c => Assert.Equal("get_weather", c.Name));
        Assert.Equal("Paris", calls[0].Arguments["city"]?.ToString());
        Assert.Equal("Zürich", calls[1].Arguments["city"]?.ToString());
        Assert.Equal(new[] { 0, 1 }, calls.Select(c => c.Index));
    }

    [Fact]
    public void AReplyCutShortKeepsWhatItHas()
    {
        var (content, thinking, _) = Parse("<|message_model|><|content_thinking|>Let me think about a <", 1000);
        Assert.Equal("", content);
        Assert.Equal("Let me think about a <", thinking);
        (content, thinking, _) = Parse("<|message_model|><|content_text|>Answer with a < sign", 1);
        Assert.Equal("Answer with a < sign", content);
        Assert.Equal("", thinking);
    }

    [Fact]
    public void TheThinkingChannelIsOpenOnlyInsideReasoning()
    {
        IOutputParser parser = NewParser();
        parser.Add("<|message_model|><|content_thinking|>hmm", done: false);
        Assert.True(parser.IsThinkingChannelOpen);
        parser.Add("<|end_message|><|message_model|><|content_text|>ok", done: false);
        Assert.False(parser.IsThinkingChannelOpen);
    }
}
