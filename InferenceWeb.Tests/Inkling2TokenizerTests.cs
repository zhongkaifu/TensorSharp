// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The real Inkling 2 Mini GGUF's tokenizer and chat template (<c>INKLING2_MODEL</c>) against the prompt ids vLLM's
/// tokenizer and chat template produced for the same texts and messages (<c>INKLING2_VLLM_REFERENCE</c>). Reads the
/// GGUF's metadata only.
/// </summary>
public sealed class Inkling2TokenizerTests
{
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Inkling2TokenizerTests(Xunit.Abstractions.ITestOutputHelper output) => _output = output;

    [ModelFact("INKLING2_MODEL")]
    public void TokenizesAndRendersAsVllmDoes()
    {
        string referencePath = Environment.GetEnvironmentVariable("INKLING2_VLLM_REFERENCE");
        Assert.False(string.IsNullOrEmpty(referencePath), "INKLING2_VLLM_REFERENCE names the vLLM reference JSON");
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(referencePath));
        using var gguf = new GgufFile(Environment.GetEnvironmentVariable("INKLING2_MODEL"));
        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        string template = gguf.GetString("tokenizer.chat_template");
        string architecture = gguf.GetString("general.architecture");

        var mismatches = new List<string>();
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int[] expected = c.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            string text;
            if (c.GetProperty("kind").GetString() == "raw")
                text = c.GetProperty("text").GetString();
            else
            {
                var messages = c.GetProperty("messages").EnumerateArray()
                    .Select(m => new ChatMessage { Role = m.GetProperty("role").GetString(), Content = m.GetProperty("content").GetString() })
                    .ToList();
                string effort = c.TryGetProperty("chat_template_kwargs", out JsonElement kwargs) && kwargs.ValueKind == JsonValueKind.Object
                    && kwargs.TryGetProperty("reasoning_effort", out JsonElement e) ? e.GetString() : null;
                text = ChatTemplate.RenderFromGgufTemplate(template, messages, addGenerationPrompt: true, architecture,
                    reasoningEffort: effort);
            }
            int[] actual = tokenizer.Encode(text, addSpecial: true).ToArray();
            if (!actual.SequenceEqual(expected))
            {
                int at = 0;
                while (at < Math.Min(actual.Length, expected.Length) && actual[at] == expected[at]) at++;
                mismatches.Add($"{c.GetProperty("kind").GetString()} case differs at token {at}: vLLM [{string.Join(",", expected.Skip(at).Take(8))}] " +
                    $"TensorSharp [{string.Join(",", actual.Skip(at).Take(8))}] in {JsonSerializer.Serialize(text)}");
            }
        }
        foreach (string m in mismatches) _output.WriteLine(m);
        Assert.Empty(mismatches);
    }
}
