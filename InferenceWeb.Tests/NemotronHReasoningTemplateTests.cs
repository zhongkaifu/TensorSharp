// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H 8B/47B Reasoning-128K ship under general.architecture = nemotron_h, the
// same name as the ChatML Nemotron 3 Nano / Omni checkpoints, but were trained on a
// <SPECIAL_10>System / <SPECIAL_11>User / <SPECIAL_11>Assistant turn format. They used
// to be rendered as ChatML, so the model saw "<|im_start|>" as plain text and answered
// with "</think>" and invented "<|im_start|>user" turns in its content.
using System.Collections.Generic;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class NemotronHReasoningTemplateTests
{
    // tokenizer.chat_template of nvidia_Nemotron-H-8B-Reasoning-128K-Q4_K_M.gguf and the
    // 47B file (byte-identical to nvidia/Nemotron-H-8B-Reasoning-128K tokenizer_config.json).
    private const string ShippedTemplate =
        "{{ '<SPECIAL_10>System\n' }}{%- if messages and messages[0]['role'] == 'system' -%}{{ messages[0]['content'].strip() }}{%- endif -%}" +
        "{% for message in (messages[1:] if messages[0]['role'] == 'system' else messages) %}{%- if message['role'] == 'user' -%}" +
        "{{ '\n<SPECIAL_11>User\n' + message['content'].strip() + '\n<SPECIAL_11>Assistant\n' }}{%- if loop.last -%}" +
        "{%- if messages[0]['role'] == 'system' -%}{%- if \"{'reasoning': True}\" in messages[0]['content'] -%}{{ '<think>\n' }}" +
        "{%- elif \"{'reasoning': False}\" in messages[0]['content'] -%}{{ '<think></think>' }}{%- endif -%}{%- endif -%}{%- endif -%}" +
        "{%- elif message['role'] == 'assistant' -%}{{ message['content'].strip() }}{%- endif -%}{%- endfor -%}";

    private static string Render(List<ChatMessage> messages, bool thinking = false, List<ToolFunction>? tools = null,
        string template = ShippedTemplate)
        => ChatTemplate.RenderFromGgufTemplate(template, messages, addGenerationPrompt: true,
            architecture: "nemotron_h", tools: tools, enableThinking: thinking);

    [Theory]
    [InlineData("nemotron_h")]
    [InlineData("nemotron_h_moe")]
    public void ShippedTemplate_IsRecognised(string arch)
    {
        Assert.True(ChatTemplate.IsNemotronHReasoningTemplate(ShippedTemplate));
        Assert.Equal("nemotron_h", ChatProtocolRegistry.For(arch)!.Id);
    }

    [Fact]
    public void ThinkingOff_RendersTrainedTurnFormat_NotChatMl()
    {
        string prompt = Render([new ChatMessage { Role = "user", Content = "What is 17 + 25?" }]);

        Assert.Equal(
            "<SPECIAL_10>System\n{'reasoning': False}\n<SPECIAL_11>User\nWhat is 17 + 25?\n<SPECIAL_11>Assistant\n<think></think>",
            prompt);
        Assert.DoesNotContain("<|im_start|>", prompt);
    }

    [Fact]
    public void ThinkingOn_OpensReasoningBlock()
    {
        string prompt = Render([new ChatMessage { Role = "user", Content = "Hi" }], thinking: true);

        Assert.Equal(
            "<SPECIAL_10>System\n{'reasoning': True}\n<SPECIAL_11>User\nHi\n<SPECIAL_11>Assistant\n<think>\n",
            prompt);
    }

    [Theory]
    [InlineData("{'reasoning': False}\nYou are terse.", false)]
    [InlineData("{'reasoning': True}", true)]
    [InlineData("{'reasoning': False}", true)] // explicit marker wins over the request flag
    public void MultiTurn_MatchesShippedJinjaTemplate(string system, bool requestThinking)
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = system },
            new() { Role = "user", Content = " What is 2+2? " },
            new() { Role = "assistant", Content = "4\n" },
            new() { Role = "user", Content = "And 3+3?" },
        };

        string expected = new Jinja2Template(ShippedTemplate).Render(new Dictionary<string, object>
        {
            ["messages"] = new List<object>
            {
                new Dictionary<string, object> { ["role"] = "system", ["content"] = system },
                new Dictionary<string, object> { ["role"] = "user", ["content"] = " What is 2+2? " },
                new Dictionary<string, object> { ["role"] = "assistant", ["content"] = "4\n" },
                new Dictionary<string, object> { ["role"] = "user", ["content"] = "And 3+3?" },
            },
            ["add_generation_prompt"] = true,
        });

        Assert.Equal(expected, Render(messages, requestThinking));
    }

    [Fact]
    public void Tools_AreDeclaredAndResultsFedBackAsUserTurn()
    {
        var tools = new List<ToolFunction>
        {
            new()
            {
                Name = "get_weather",
                Description = "Weather for a city",
                Parameters = new Dictionary<string, ToolParameter> { ["city"] = new() { Type = "string" } },
            },
        };
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Weather in Paris?" },
            new()
            {
                Role = "assistant",
                Content = "",
                ToolCalls = [new ToolCall { Name = "get_weather", Arguments = new Dictionary<string, object> { ["city"] = "Paris" } }],
            },
            new() { Role = "tool", Content = "{\"temp\": 21}" },
        };

        string prompt = Render(messages, tools: tools);

        Assert.StartsWith("<SPECIAL_10>System\n{'reasoning': False}\n\n# Tools", prompt);
        Assert.Contains("\"name\": \"get_weather\"", prompt);
        Assert.Contains(
            "\n<SPECIAL_11>Assistant\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>" +
            "\n<SPECIAL_11>User\n<tool_response>\n{\"temp\": 21}\n</tool_response>\n<SPECIAL_11>Assistant\n<think></think>",
            prompt);
        Assert.DoesNotContain("<|im_start|>", prompt);
    }

    [Fact]
    public void ChatMlNemotronTemplate_StillRendersChatMl()
    {
        const string chatMl = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}";
        string prompt = Render([new ChatMessage { Role = "user", Content = "Hi" }], template: chatMl);

        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>", prompt);
        Assert.False(ChatTemplate.IsNemotronHReasoningTemplate(chatMl));
        Assert.False(ChatTemplate.IsNemotronHReasoningTemplate(null));
    }

    /// <summary>The tool-call body is model output. Nemotron-H 8B at concurrency 4 answered
    /// with a JSON list inside &lt;tool_call&gt; (its own tool format is a list), and
    /// <c>GetProperty("name")</c> on the array threw InvalidOperationException, which
    /// aborted the streamed HTTP response mid-flight. No JSON shape may throw; a list of
    /// call objects yields each call.</summary>
    [Theory]
    [InlineData("<tool_call>[{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}]</tool_call>", 1)]
    [InlineData("<tool_call>[{\"name\":\"a\",\"arguments\":{}},{\"name\":\"b\",\"arguments\":{}}]</tool_call>", 2)]
    [InlineData("<tool_call>[1, \"x\", null]</tool_call>", 0)]
    [InlineData("<tool_call>\"get_weather\"</tool_call>", 0)]
    [InlineData("<tool_call>{\"arguments\":{\"city\":\"Paris\"}}</tool_call>", 0)]
    [InlineData("<tool_call>{\"name\":42,\"arguments\":{}}</tool_call>", 0)]
    [InlineData("<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>", 1)]
    public void ChatMlParser_ToolCallBodyOfAnyJsonShape_DoesNotThrow(string output, int expectedCalls)
    {
        var parser = new ChatMlOutputParser();
        parser.Init(enableThinking: false, tools: null);

        ParsedOutput parsed = parser.Add(output, done: true);

        Assert.Equal(expectedCalls, parsed.ToolCalls?.Count ?? 0);
        if (expectedCalls > 0)
            Assert.All(parsed.ToolCalls!, call => Assert.False(string.IsNullOrEmpty(call.Name)));
    }
}
