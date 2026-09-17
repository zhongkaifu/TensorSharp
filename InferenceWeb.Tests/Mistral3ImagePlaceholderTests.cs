// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Mistral 3 renders through its own template (PreferOwnRenderer), which skipped the
// generic media-placeholder pass: an image request produced a prompt with no [IMG],
// ModelMultimodalInjector found nothing to expand, and the encoded image never reached
// the model (Mistral Small 3.1 read a red "4821" card as a blue "2975").
using System.Collections.Generic;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Mistral3ImagePlaceholderTests
{
    [Fact]
    public void ServerRenderPath_EmitsOneImgPlaceholderPerImage_BeforeTheText()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Answer in JSON." },
            new() { Role = "user", Content = "Read both codes.", ImagePaths = new List<string> { "a.png", "b.png" } },
        };

        string prompt = ChatTemplate.RenderFromGgufTemplate("", messages, true, "mistral3");

        Assert.Equal("[SYSTEM_PROMPT]Answer in JSON.[/SYSTEM_PROMPT][INST][IMG][IMG]Read both codes.[/INST]", prompt);
    }

    [Fact]
    public void FollowUpTurn_KeepsTheEarlierImagePlaceholder_AndTextOnlyTurnsGetNone()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is the code?", ImagePaths = new List<string> { "card.png" } },
            new() { Role = "assistant", Content = "4821" },
            new() { Role = "user", Content = "Repeat it as JSON." },
        };

        string prompt = ChatTemplate.RenderMistral3(messages);

        Assert.Equal(1, Regex.Matches(prompt, Regex.Escape("[IMG]")).Count);
        Assert.StartsWith("[INST][IMG]What is the code?[/INST]4821[INST]Repeat it as JSON.[/INST]", prompt);
    }
}
