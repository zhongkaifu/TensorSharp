// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3-VL's chat protocol. Its Instruct checkpoints' template has no reasoning block, so the generation prompt must end
// "<|im_start|>assistant\n" exactly as the template renders it: Qwen 3.5's protocol (which these architectures used to
// share) closed an empty <think> block on every thinking-off prompt, and the renderer's trailing-whitespace trim took the
// newline, after which the model opened its answers with ": " or a blank line.
using System.Collections.Generic;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Qwen3VLChatProtocolTests
{
    // The generation-prompt part of Qwen3-VL-30B-A3B-Instruct's shipped template (user and assistant turns, no tools).
    private const string InstructTemplate =
        "{%- for message in messages %}" +
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}" +
        "{%- endfor %}" +
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}";

    // A Thinking checkpoint's template opens the block after the assistant marker.
    private const string ThinkingTemplate =
        "{%- for message in messages %}" +
        "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}" +
        "{%- endfor %}" +
        "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n' }}{%- endif %}";

    [Theory]
    [InlineData("qwen3vl")]
    [InlineData("qwen3vlmoe")]
    public void BothArchitecturesUseTheQwen3VLProtocol(string architecture)
    {
        ChatProtocol protocol = ChatProtocolRegistry.For(architecture)!;
        Assert.Equal("qwen3vl", protocol.Id);
        Assert.True(protocol.CapsVideoFrames);
        Assert.True(protocol.KeepsGenerationPromptWhitespace);
        Assert.Equal("qwen35", ChatProtocolRegistry.For("qwen35moe")!.Id);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnInstructTemplateEndsAtTheAssistantMarkerWithItsNewline(bool thinking)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Describe the clip." } };
        string prompt = ChatTemplate.RenderFromGgufTemplate(InstructTemplate, messages, true, "qwen3vlmoe", enableThinking: thinking);
        Assert.EndsWith("<|im_start|>assistant\n", prompt);
        Assert.DoesNotContain("<think>", prompt);
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3vlmoe", thinking, InstructTemplate));
    }

    [Fact]
    public void AThinkingTemplateKeepsItsOpenBlock()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Describe the clip." } };
        string prompt = ChatTemplate.RenderFromGgufTemplate(ThinkingTemplate, messages, true, "qwen3vlmoe");
        Assert.EndsWith("<|im_start|>assistant\n<think>\n", prompt);
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3vlmoe", true, ThinkingTemplate));
    }

    [Fact]
    public void HistoryWithoutAGenerationPromptIsStillTrimmed()
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "Hi." } };
        string prompt = ChatTemplate.RenderFromGgufTemplate(InstructTemplate, messages, false, "qwen3vlmoe");
        Assert.EndsWith("<|im_end|>", prompt);
    }
}
