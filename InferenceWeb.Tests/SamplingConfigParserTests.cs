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
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Verifies that the per-request samplers correctly merge defaults with body
/// overrides. The interesting cases are:
///   - body overrides win over defaults (per-request control)
///   - defaults are returned untouched when the body says nothing (server config)
///   - the parser never mutates the supplied defaults instance (so a singleton
///     can be shared across requests safely)
/// </summary>
public class SamplingConfigParserTests
{
    [Fact]
    public void ParseWebUi_NoOverrides_ReturnsCloneOfDefaults()
    {
        var defaults = new SamplingConfig
        {
            Temperature = 0.42f,
            TopK = 7,
            TopP = 0.55f,
            MinP = 0.11f,
            RepetitionPenalty = 1.35f,
            PresencePenalty = 0.5f,
            FrequencyPenalty = 0.25f,
            Seed = 1234,
        };

        using var doc = JsonDocument.Parse("{}");
        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults);

        Assert.Equal(0.42f, parsed.Temperature);
        Assert.Equal(7, parsed.TopK);
        Assert.Equal(0.55f, parsed.TopP);
        Assert.Equal(0.11f, parsed.MinP);
        Assert.Equal(1.35f, parsed.RepetitionPenalty);
        Assert.Equal(0.5f, parsed.PresencePenalty);
        Assert.Equal(0.25f, parsed.FrequencyPenalty);
        Assert.Equal(1234, parsed.Seed);
        // Cloned not aliased: mutating the parse result must not bleed back
        // into the shared defaults instance.
        Assert.NotSame(defaults, parsed);
    }

    [Fact]
    public void ParseWebUi_WithOverrides_RequestValuesWinOverDefaults()
    {
        var defaults = new SamplingConfig { Temperature = 0.1f, TopK = 5, TopP = 0.3f };
        using var doc = JsonDocument.Parse("""{"temperature": 0.9, "top_k": 64, "top_p": 0.95}""");

        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults);

        Assert.Equal(0.9f, parsed.Temperature);
        Assert.Equal(64, parsed.TopK);
        Assert.Equal(0.95f, parsed.TopP);
    }

    [Fact]
    public void ParseWebUi_PartialOverrides_KeepsDefaultsForUntouchedFields()
    {
        var defaults = new SamplingConfig
        {
            Temperature = 0.3f,
            TopK = 20,
            TopP = 0.7f,
            MinP = 0.05f,
            RepetitionPenalty = 1.2f,
        };
        using var doc = JsonDocument.Parse("""{"temperature": 1.5}""");

        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults);

        Assert.Equal(1.5f, parsed.Temperature);
        Assert.Equal(20, parsed.TopK);
        Assert.Equal(0.7f, parsed.TopP);
        Assert.Equal(0.05f, parsed.MinP);
        Assert.Equal(1.2f, parsed.RepetitionPenalty);
    }

    [Fact]
    public void ParseWebUi_AcceptsCamelCaseAndSnakeCase()
    {
        var defaults = new SamplingConfig();
        using var doc = JsonDocument.Parse("""
            {"topK": 20, "min_p": 0.07, "repetitionPenalty": 1.25, "frequency_penalty": 0.4}
            """);

        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults);

        Assert.Equal(20, parsed.TopK);
        Assert.Equal(0.07f, parsed.MinP);
        Assert.Equal(1.25f, parsed.RepetitionPenalty);
        Assert.Equal(0.4f, parsed.FrequencyPenalty);
    }

    [Fact]
    public void ParseOllama_ReadsValuesFromOptionsObject()
    {
        var defaults = new SamplingConfig { Temperature = 0.2f, TopK = 5 };
        using var doc = JsonDocument.Parse("""
            {"options": {"temperature": 0.85, "top_k": 50, "repeat_penalty": 1.4, "stop": ["</s>"]}}
            """);

        var parsed = SamplingConfigParser.ParseOllama(doc.RootElement, defaults);

        Assert.Equal(0.85f, parsed.Temperature);
        Assert.Equal(50, parsed.TopK);
        Assert.Equal(1.4f, parsed.RepetitionPenalty);
        Assert.NotNull(parsed.StopSequences);
        Assert.Single(parsed.StopSequences, "</s>");
    }

    [Fact]
    public void ParseOllama_NoOptionsObject_KeepsDefaults()
    {
        var defaults = new SamplingConfig { Temperature = 0.6f, TopK = 25 };
        using var doc = JsonDocument.Parse("{}");

        var parsed = SamplingConfigParser.ParseOllama(doc.RootElement, defaults);

        Assert.Equal(0.6f, parsed.Temperature);
        Assert.Equal(25, parsed.TopK);
    }

    [Fact]
    public void ParseOpenAI_AcceptsStopAsSingleString()
    {
        var defaults = new SamplingConfig();
        using var doc = JsonDocument.Parse("""
            {"temperature": 0.5, "top_p": 0.8, "stop": "<|eot|>"}
            """);

        var parsed = SamplingConfigParser.ParseOpenAI(doc.RootElement, defaults);

        Assert.Equal(0.5f, parsed.Temperature);
        Assert.Equal(0.8f, parsed.TopP);
        Assert.NotNull(parsed.StopSequences);
        Assert.Single(parsed.StopSequences, "<|eot|>");
    }

    [Fact]
    public void ParseOpenAI_AcceptsStopAsArray()
    {
        var defaults = new SamplingConfig();
        using var doc = JsonDocument.Parse("""
            {"stop": ["</s>", "<|eot|>"]}
            """);

        var parsed = SamplingConfigParser.ParseOpenAI(doc.RootElement, defaults);

        Assert.Equal(new[] { "</s>", "<|eot|>" }, parsed.StopSequences);
    }

    [Fact]
    public void ParseWebUi_NullDefaults_FallsBackToBuiltInDefaults()
    {
        var fallback = new SamplingConfig();
        using var doc = JsonDocument.Parse("{}");

        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults: null);

        // No body overrides, no caller defaults: we should land on the
        // SamplingConfig type's built-in defaults (matches Ollama: 0.8/40/0.9).
        Assert.Equal(fallback.Temperature, parsed.Temperature);
        Assert.Equal(fallback.TopK, parsed.TopK);
        Assert.Equal(fallback.TopP, parsed.TopP);
    }

    [Fact]
    public void ParseWebUi_DoesNotMutateProvidedDefaults()
    {
        var defaults = new SamplingConfig
        {
            Temperature = 0.42f,
            StopSequences = new List<string> { "ALPHA" },
        };
        using var doc = JsonDocument.Parse("""
            {"temperature": 0.99, "stop": ["BETA"]}
            """);

        var parsed = SamplingConfigParser.ParseWebUi(doc.RootElement, defaults);

        // The parse must have produced new values without touching the
        // singleton defaults instance.
        Assert.Equal(0.99f, parsed.Temperature);
        Assert.Equal(new[] { "BETA" }, parsed.StopSequences);
        Assert.Equal(0.42f, defaults.Temperature);
        Assert.Equal(new[] { "ALPHA" }, defaults.StopSequences);
        Assert.NotSame(defaults.StopSequences, parsed.StopSequences);
    }
}
