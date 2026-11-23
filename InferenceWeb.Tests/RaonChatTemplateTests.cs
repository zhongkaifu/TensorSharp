// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>Raon's generation prompt keeps the newlines its template ends with ("&lt;think&gt;\n", "&lt;think&gt;&lt;/think&gt;\n\n"),
/// which the renderer's blanket TrimEnd would take: without them the model opens its answer with them itself. The template is
/// the generation-prompt tail of the GGUF's (RaonParityTests checks the whole one against Transformers).</summary>
public sealed class RaonChatTemplateTests
{
    private const string Template =
        "{%- for message in messages %}{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}{%- endfor %}"
        + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
        + "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think></think>\\n\\n' }}"
        + "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}";

    [Theory]
    [InlineData(false, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think></think>\n\n")]
    [InlineData(true, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n")]
    public void GenerationPromptKeepsItsTrailingNewlines(bool thinking, string expected)
        => Assert.Equal(expected, ChatTemplate.RenderFromGgufTemplate(Template,
            new List<ChatMessage> { new() { Role = "user", Content = "Hi" } }, true, "raon1nano", enableThinking: thinking));
}
