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
// Unit tests for gpt-oss / Harmony function (tool) calling: the hardcoded prompt
// renderer (ChatTemplate.RenderHarmony) and the streaming output parser
// (HarmonyOutputParser). These run without a model and validate the exact wire
// format against the official gpt-oss chat template's conventions.

using System.Collections.Generic;

namespace InferenceWeb.Tests;

public class HarmonyToolCallTests
{
    private static List<ToolFunction> WeatherTool() => new()
    {
        new ToolFunction
        {
            Name = "get_weather",
            Description = "Get the current weather in a city",
            Parameters = new Dictionary<string, ToolParameter>
            {
                ["location"] = new ToolParameter { Type = "string", Description = "The city name" },
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

    // ---- Rendering -------------------------------------------------------

    [Fact]
    public void Render_WithTools_EmitsCommentaryChannelLineAndToolNamespace()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What's the weather in Paris?" },
        };

        string prompt = ChatTemplate.RenderHarmony(messages, addGenerationPrompt: true, tools: WeatherTool());

        // System message advertises the functions tool channel only when tools exist.
        Assert.Contains("# Valid channels: analysis, commentary, final.", prompt);
        Assert.Contains("Calls to these tools must go to the commentary channel: 'functions'.", prompt);

        // Developer message carries the TypeScript-style tool namespace.
        Assert.Contains("<|start|>developer<|message|>", prompt);
        Assert.Contains("# Tools\n\n## functions\n\nnamespace functions {", prompt);
        Assert.Contains("// Get the current weather in a city", prompt);
        Assert.Contains("type get_weather = (_: {", prompt);
        Assert.Contains("// The city name\nlocation: string,", prompt);     // required => no '?'
        Assert.Contains("unit?: \"celsius\" | \"fahrenheit\",", prompt);    // optional + enum union
        Assert.Contains("}) => any;", prompt);
        Assert.Contains("} // namespace functions", prompt);

        // Generation prompt.
        Assert.EndsWith("<|start|>assistant", prompt);
    }

    [Fact]
    public void Render_WithoutTools_OmitsToolChannelLineAndNamespace()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hello" },
        };

        string prompt = ChatTemplate.RenderHarmony(messages, addGenerationPrompt: true, tools: null);

        Assert.DoesNotContain("Calls to these tools must go to the commentary channel", prompt);
        Assert.DoesNotContain("# Tools", prompt);
        Assert.DoesNotContain("namespace functions", prompt);
        Assert.Contains("<|start|>user<|message|>Hello<|end|>", prompt);
        Assert.EndsWith("<|start|>assistant", prompt);
    }

    [Fact]
    public void Render_NoParameterTool_UsesArrowAnyForm()
    {
        var tools = new List<ToolFunction>
        {
            new ToolFunction { Name = "get_time", Description = "Get the current time" },
        };
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "time?" } };

        string prompt = ChatTemplate.RenderHarmony(messages, true, tools);

        Assert.Contains("type get_time = () => any;", prompt);
    }

    [Fact]
    public void Render_AssistantToolCallAndToolResult_RoundTripsFraming()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Weather in Paris?" },
            new()
            {
                Role = "assistant",
                Thinking = "The user wants weather; call get_weather.",
                ToolCalls = new List<ToolCall>
                {
                    new ToolCall
                    {
                        Name = "get_weather",
                        Arguments = new Dictionary<string, object> { ["location"] = "Paris" },
                    },
                },
            },
            new() { Role = "tool", Content = "{\"temperature\": 18, \"unit\": \"celsius\"}" },
        };

        string prompt = ChatTemplate.RenderHarmony(messages, addGenerationPrompt: true, tools: WeatherTool());

        // Reasoning replayed on the analysis channel for the active tool-call turn.
        Assert.Contains("<|start|>assistant<|channel|>analysis<|message|>The user wants weather; call get_weather.<|end|>", prompt);
        // The assistant tool call on the commentary channel, terminated by <|call|>.
        Assert.Contains("<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"location\":\"Paris\"}<|call|>", prompt);
        // The tool result attributed back to the function, addressed to the assistant.
        Assert.Contains("<|start|>functions.get_weather to=assistant<|channel|>commentary<|message|>{\"temperature\": 18, \"unit\": \"celsius\"}<|end|>", prompt);
    }

    // ---- Parsing ---------------------------------------------------------

    [Fact]
    public void Parser_AdvertisesToolSupport()
    {
        var parser = new HarmonyOutputParser();
        Assert.True(parser.HasToolSupport);
        Assert.True(parser.HasThinkingSupport);
        Assert.True(parser.AlwaysRequired);
    }

    [Fact]
    public void Parser_AnalysisThenToolCall_ExtractsThinkingAndCall()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: WeatherTool());

        // What the model emits after the prompt's "<|start|>assistant" generation
        // marker. <|call|> is a stop token (dropped), so generation ends with
        // done=true after the JSON arguments.
        string modelOut =
            "<|channel|>analysis<|message|>Need the weather for SF.<|end|>" +
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json" +
            "<|message|>{\"location\":\"San Francisco\",\"days\":3}";

        var parsed = parser.Add(modelOut, done: true);

        Assert.Equal("Need the weather for SF.", parsed.Thinking);
        Assert.Equal("", parsed.Content);
        Assert.NotNull(parsed.ToolCalls);
        Assert.Single(parsed.ToolCalls!);

        var call = parsed.ToolCalls![0];
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("San Francisco", call.Arguments["location"]);
        Assert.Equal(3L, call.Arguments["days"]);   // integers parse to long
    }

    [Fact]
    public void Parser_ToolCallWithExplicitCallTag_AlsoExtracts()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: false, tools: WeatherTool());

        // Defensive: even if <|call|> arrives as text (not consumed as a stop token).
        string modelOut =
            "<|channel|>commentary to=functions.get_weather <|constrain|>json" +
            "<|message|>{\"location\":\"Tokyo\"}<|call|>";

        var parsed = parser.Add(modelOut, done: true);

        Assert.NotNull(parsed.ToolCalls);
        Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", parsed.ToolCalls![0].Name);
        Assert.Equal("Tokyo", parsed.ToolCalls![0].Arguments["location"]);
    }

    [Fact]
    public void Parser_PlainFinalResponse_NoToolCalls()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: new List<ToolFunction>());

        string modelOut =
            "<|channel|>analysis<|message|>Simple greeting.<|end|>" +
            "<|start|>assistant<|channel|>final<|message|>Hello there!";

        var parsed = parser.Add(modelOut, done: true);

        Assert.Equal("Simple greeting.", parsed.Thinking);
        Assert.Equal("Hello there!", parsed.Content);
        Assert.Null(parsed.ToolCalls);
    }

    [Fact]
    public void Parser_StreamedInChunks_StillExtractsToolCall()
    {
        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: WeatherTool());

        // Split at safe boundaries (special tokens stay intact; the JSON body is
        // split mid-string to exercise tool-argument accumulation across Add calls).
        string[] chunks =
        {
            "<|channel|>analysis<|message|>",
            "thinking...",
            "<|end|>",
            "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>",
            "{\"loca",
            "tion\":\"Berlin\"}",
        };

        ParsedOutput last = null!;
        string thinking = "";
        for (int i = 0; i < chunks.Length; i++)
        {
            last = parser.Add(chunks[i], done: i == chunks.Length - 1);
            thinking += last.Thinking;
        }

        Assert.Equal("thinking...", thinking);
        Assert.NotNull(last.ToolCalls);
        Assert.Single(last.ToolCalls!);
        Assert.Equal("get_weather", last.ToolCalls![0].Name);
        Assert.Equal("Berlin", last.ToolCalls![0].Arguments["location"]);
    }

    [Fact]
    public void RenderThenParse_RoundTrip_RecoversToolCall()
    {
        // Render a tool-enabled prompt (sanity that it has the namespace), then feed
        // a representative model completion through the parser and recover the call.
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Weather in Rome?" } };
        string prompt = ChatTemplate.RenderHarmony(messages, true, WeatherTool());
        Assert.EndsWith("<|start|>assistant", prompt);

        var parser = new HarmonyOutputParser();
        parser.Init(enableThinking: true, tools: WeatherTool());
        var parsed = parser.Add(
            "<|channel|>commentary to=functions.get_weather <|constrain|>json" +
            "<|message|>{\"location\":\"Rome\",\"unit\":\"celsius\"}",
            done: true);

        Assert.NotNull(parsed.ToolCalls);
        var call = Assert.Single(parsed.ToolCalls!);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Rome", call.Arguments["location"]);
        Assert.Equal("celsius", call.Arguments["unit"]);
    }
}
