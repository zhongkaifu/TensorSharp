// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// DiffusionGemma denoises whole canvases written in Gemma 4's channel syntax. It had
/// no chat protocol, so the raw canvas - <c>&lt;|channel&gt;thought</c> marker and all -
/// was the OpenAI answer (38/39 JSON checks failed in the release campaign). No real
/// model here: these pin the protocol entry and the channel separation the pipeline
/// applies to every preview and to the final text.
/// </summary>
public class DiffusionGemmaProtocolTests
{
    [Fact]
    public void BothArchitectureSpellings_ShareAMandatoryGemma4Parser()
    {
        var protocol = ChatProtocolRegistry.For("diffusion-gemma");
        Assert.NotNull(protocol);
        Assert.Same(protocol, ChatProtocolRegistry.For("diffusion_gemma"));
        Assert.True(protocol!.OutputParserAlwaysRequired);
        Assert.True(OutputParserFactory.IsAlwaysRequired("diffusion-gemma"));
        Assert.IsType<Gemma4OutputParser>(OutputParserFactory.Create("diffusion-gemma"));
        // The GGUF template keeps rendering the prompt.
        Assert.Null(protocol.Render);
        Assert.Null(protocol.PreferOwnRenderer);
    }

    [Fact]
    public void RegisteringTheParser_DoesNotMakeTheFamilyToolCapable()
    {
        // Before this entry existed the passthrough parser kept ToolsRendered false,
        // so --code-exec and skills discovery never offered a diffusion request the
        // shell / skills_read tools (the denoising pipeline renders with tools: null
        // and refuses client tools). Gemma4OutputParser CAN read a call back, so the
        // protocol has to say explicitly that declarations never reach the prompt,
        // or every adapter starts leasing a workspace and running the skills loop
        // over a model that was never told about the tools.
        Assert.False(ChatProtocolRegistry.For("diffusion-gemma")!.RendersToolDeclarations);
        Assert.False(SkillCapabilities.For("diffusion-gemma").ToolsRendered);
        Assert.False(SkillCapabilities.For("diffusion_gemma").ToolsRendered);
    }

    [Fact]
    public void OpenedThoughtChannel_IsDroppedUnlessRequested()
    {
        const string raw = "<|channel>thought\nLet me add.<channel|>{\n  \"answer\": 42\n}";

        var (content, thinking) = ChatGenerationPipeline.SeparateDiffusionChannels(
            "diffusion-gemma", raw, enableThinking: false, generationSuffix: null);
        Assert.Equal("{\n  \"answer\": 42\n}", content);
        Assert.Null(thinking);

        (content, thinking) = ChatGenerationPipeline.SeparateDiffusionChannels(
            "diffusion-gemma", raw, enableThinking: true, generationSuffix: null);
        Assert.Equal("{\n  \"answer\": 42\n}", content);
        Assert.Equal("Let me add.", thinking);
    }

    [Fact]
    public void StrayChannelClose_AndPromptPrimedChannel_AreNotContent()
    {
        // The Mac campaign's shape: the model closes a channel the prompt opened.
        var (content, _) = ChatGenerationPipeline.SeparateDiffusionChannels(
            "diffusion-gemma", "<channel|>{\"tag\": \"x\"}", enableThinking: false, generationSuffix: null);
        Assert.Equal("{\"tag\": \"x\"}", content);

        // With the primer known, the text before the close is the thought.
        var (primedContent, primedThinking) = ChatGenerationPipeline.SeparateDiffusionChannels(
            "diffusion-gemma", "17 + 25 = 42<channel|>{\"answer\": 42}", enableThinking: true,
            generationSuffix: "<|channel>thought\n");
        Assert.Equal("{\"answer\": 42}", primedContent);
        Assert.Equal("17 + 25 = 42", primedThinking);
    }

    [Fact]
    public void PlainText_AndEmptyCanvas_PassThrough()
    {
        Assert.Equal(("Hello! How can I help you today?", (string?)null),
            ChatGenerationPipeline.SeparateDiffusionChannels(
                "diffusion-gemma", "Hello! How can I help you today?", enableThinking: false, generationSuffix: null));
        Assert.Equal((string.Empty, (string?)null),
            ChatGenerationPipeline.SeparateDiffusionChannels("diffusion-gemma", "", false, null));
    }

    [Fact]
    public void ASpontaneousToolCall_SurfacesAsText_BecauseNothingCanServiceIt()
    {
        var (content, _) = ChatGenerationPipeline.SeparateDiffusionChannels(
            "diffusion-gemma", "<|tool_call>call:probe{}<tool_call|>", enableThinking: false, generationSuffix: null);
        Assert.Equal("call:probe{}", content);
    }
}
