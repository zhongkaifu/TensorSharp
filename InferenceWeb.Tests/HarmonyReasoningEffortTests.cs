// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorSharp.Runtime;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// GPT-OSS (Harmony) always reasons in its analysis channel; the only lever over
/// how long is the system message's <c>Reasoning: low|medium|high</c> line, which
/// used to be hard-coded to <c>medium</c>. These pin the OpenAI <c>reasoning_effort</c>
/// field, its mapping from an explicit <c>think:false</c>, and the delayed grammar
/// trigger that lets <c>response_format</c> combine with <c>think:true</c>.
/// </summary>
public class HarmonyReasoningEffortTests
{
    private static readonly List<ChatMessage> Ask = new() { new() { Role = "user", Content = "Hi" } };

    [Fact]
    public void Harmony_RendersTheRequestedLevel_AndDefaultsToMedium()
    {
        Assert.Contains("\nReasoning: medium\n\n", ChatTemplate.RenderHarmony(Ask), StringComparison.Ordinal);
        Assert.Contains("\nReasoning: low\n\n", ChatTemplate.RenderHarmony(Ask, reasoningEffort: "low"), StringComparison.Ordinal);
        Assert.Contains("\nReasoning: high\n\n", ChatTemplate.RenderHarmony(Ask, reasoningEffort: "High"), StringComparison.Ordinal);
        // An unknown spelling never reaches the renderer (the adapters answer 400), but
        // the renderer itself falls back rather than printing garbage into the prompt.
        Assert.Contains("\nReasoning: medium\n\n", ChatTemplate.RenderHarmony(Ask, reasoningEffort: "max"), StringComparison.Ordinal);
    }

    [Fact]
    public void TheLevel_ReachesHarmonyThroughTheGgufTemplateEntryPoint()
    {
        // Harmony always prefers its own renderer, so the level must survive the
        // generic RenderFromGgufTemplate path the KV-cache renderer calls.
        string prompt = ChatTemplate.RenderFromGgufTemplate(
            "{{ messages }}", Ask, addGenerationPrompt: true, architecture: "gpt-oss", reasoningEffort: "high");
        Assert.Contains("\nReasoning: high\n\n", prompt, StringComparison.Ordinal);
        Assert.EndsWith("<|start|>assistant", prompt, StringComparison.Ordinal);

        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        Assert.True(ChatProtocolRegistry.For("gpt-oss")!.RendersReasoningEffort);
        Assert.False(ChatProtocolRegistry.For("gemma4")!.RendersReasoningEffort);
    }

    [Theory]
    [InlineData(null, false, null)]
    [InlineData(null, true, "low")]
    [InlineData("", true, "low")]
    [InlineData("high", true, "high")]
    [InlineData(" MEDIUM ", false, "medium")]
    public void ForRequest_MapsAnExplicitThinkFalseToLow_UnlessAnEffortWasNamed(string? requested, bool thinkFalse, string? expected)
    {
        Assert.Equal(expected, ReasoningEffort.ForRequest(requested, thinkFalse));
    }

    [Fact]
    public void TryNormalize_RejectsUnknownLevels_AndTreatsEmptyAsAbsent()
    {
        Assert.True(ReasoningEffort.TryNormalize(null, out string? none));
        Assert.Null(none);
        Assert.True(ReasoningEffort.TryNormalize("Low", out string? low));
        Assert.Equal("low", low);
        Assert.False(ReasoningEffort.TryNormalize("max", out _));
        Assert.False(ReasoningEffort.TryNormalize("none", out _));
    }

    [Theory]
    [InlineData("{\"reasoning_effort\":\"low\"}", "low")]
    [InlineData("{\"reasoning_effort\":\"HIGH\",\"think\":false}", "high")]
    [InlineData("{\"think\":false}", "low")]
    [InlineData("{\"think\":true}", null)]
    [InlineData("{}", null)]
    [InlineData("{\"reasoning_effort\":null,\"think\":false}", "low")]
    [InlineData("{\"reasoning\":{\"effort\":\"high\"}}", "high")]
    public void Parser_ReadsBothSpellings_AndTheThinkFalseMapping(string body, string? expected)
    {
        using var doc = JsonDocument.Parse(body);
        Assert.True(ReasoningEffortParser.TryParse(doc.RootElement, out string? effort, out string? error), error);
        Assert.Equal(expected, effort);
    }

    [Theory]
    [InlineData("{\"reasoning_effort\":\"max\"}")]
    [InlineData("{\"reasoning_effort\":3}")]
    [InlineData("{\"reasoning\":{\"effort\":\"minimal\"}}")]
    public void Parser_RefusesOtherValues(string body)
    {
        using var doc = JsonDocument.Parse(body);
        Assert.False(ReasoningEffortParser.TryParse(doc.RootElement, out _, out string? error));
        Assert.Contains("low", error, StringComparison.Ordinal);
    }

    [Fact]
    public void SamplingConfigClone_CarriesTheLevel()
    {
        var cfg = new SamplingConfig { ReasoningEffort = "low" };
        Assert.Equal("low", cfg.Clone().ReasoningEffort);
    }

    [Fact]
    public void HarmonyDeclaresTheSameGrammarTriggerForThinkingRequests()
    {
        // The final channel opens with this header whether or not the request asked
        // for thinking; the structured-output check needs the thinking trigger to be
        // declared, or it refuses response_format with think=true.
        var protocol = ChatProtocolRegistry.For("gpt-oss")!;
        Assert.Equal("final<|message|>", protocol.GrammarActivationTrigger);
        Assert.Equal("final<|message|>", protocol.ThinkingGrammarActivationTrigger);
        Assert.Same(protocol, ChatProtocolRegistry.For("gptoss"));
    }
}
