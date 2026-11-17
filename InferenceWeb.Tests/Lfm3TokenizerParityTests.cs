// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM2 / LFM3 tokenization (tokenizer.ggml.pre "lfm2": llama.cpp's LLAMA3 split with ignore_merges) against
/// llama.cpp. Fixtures/Lfm3Tokenizer/llamacpp-reference.json holds the ids llama-tokenize gives an LFM3 GGUF
/// (special tokens parsed, no BOS) for digits, contractions, whitespace runs, CJK and other scripts, emoji and
/// supplementary-plane text, code, punctuation runs and the chat framing's special tokens. Regenerate with
/// eng/lfm3/tokenizer-fixture.py.
/// </summary>
public class Lfm3TokenizerParityTests
{
    private sealed record Case(string Text, int[] Ids);

    private static IReadOnlyList<Case> Cases()
    {
        using var stream = typeof(Lfm3TokenizerParityTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Lfm3Tokenizer.llamacpp-reference.json")!;
        using var doc = JsonDocument.Parse(stream);
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
            c.GetProperty("text").GetString()!,
            c.GetProperty("ids").EnumerateArray().Select(p => p.GetInt32()).ToArray())).ToList();
    }

    /// <summary>The LLAMA3 split keeps digit runs of up to three and attaches one leading non-letter to a word; the
    /// default split (what "lfm2" fell through to) took every digit alone.</summary>
    [Theory]
    [InlineData("12345", new[] { "123", "45" })]
    [InlineData("2026-11-16", new[] { "202", "6", "-", "11", "-", "16" })]
    [InlineData("I'm don't", new[] { "I", "'m", " don", "'t" })]
    [InlineData("a  b\n\nc", new[] { "a", " ", " b", "\n\n", "c" })]
    [InlineData("x = 1;", new[] { "x", " =", " ", "1", ";" })]
    public void PreTokenizer_SplitsAsLlama3(string text, string[] pieces)
    {
        var tokenizer = new BpeTokenizer([], [], [], -1, [], false, false, "lfm2");
        Assert.Equal(pieces, tokenizer.SplitForBpe(text));
    }

    /// <summary>End to end through the production GGUF dispatch on an LFM3 GGUF (any quant: the tokenizer is the
    /// same): every case encodes to llama.cpp's ids.</summary>
    [ModelFact("TS_TEST_LFM3_GGUF")]
    public void Encode_FromLfm3Gguf_MatchesLlamaCppIds()
    {
        using var gguf = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable("TS_TEST_LFM3_GGUF")!);
        Assert.Equal("lfm2", gguf.GetString("tokenizer.ggml.pre", null));
        var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var cases = Cases();
        Assert.True(cases.Count >= 70);
        var mismatches = cases
            .Where(c => !c.Ids.SequenceEqual(tokenizer.Encode(c.Text, addSpecial: false)))
            .Select(c => $"{JsonSerializer.Serialize(c.Text)}: expected [{string.Join(",", c.Ids)}], " +
                         $"got [{string.Join(",", tokenizer.Encode(c.Text, addSpecial: false))}]")
            .ToList();
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }
}
