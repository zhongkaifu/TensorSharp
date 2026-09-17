// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;

namespace InferenceWeb.Tests;

/// <summary>
/// GLM-4 / GLM-5.x pre-tokenizer (tokenizer.ggml.pre "glm4" or "chatglm-bpe") against
/// the reference tokenizer. Fixtures/Glm4Tokenizer/reference.json was generated from
/// zai-org/GLM-5's tokenizer.json (MIT) with HF tokenizers: <c>pieces</c> are the
/// pre-tokenizer's offsets and <c>ids</c> are <c>encode(add_special_tokens=False)</c>
/// over numbers, CJK, code, whitespace runs, contractions and supplementary-plane
/// text. That vocabulary and merge list are identical to the GLM-5.2, GLM-5.3 and
/// GLM-5.3-Flash GGUFs.
/// </summary>
public class Glm4TokenizerParityTests
{
    private sealed record Case(string Text, string[] Pieces, int[] Ids);

    private static IReadOnlyList<Case> Cases()
    {
        using var stream = typeof(Glm4TokenizerParityTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Glm4Tokenizer.reference.json")!;
        using var doc = JsonDocument.Parse(stream);
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new Case(
            c.GetProperty("text").GetString()!,
            c.GetProperty("pieces").EnumerateArray().Select(p => p.GetString()!).ToArray(),
            c.GetProperty("ids").EnumerateArray().Select(p => p.GetInt32()).ToArray())).ToList();
    }

    [Theory]
    [InlineData("glm4")]
    [InlineData("chatglm-bpe")]
    public void PreTokenizer_MatchesTheReferenceSplitOnEveryCase(string pre)
    {
        var tokenizer = new BpeTokenizer([], [], [], -1, [], false, false, pre);
        var cases = Cases();
        Assert.True(cases.Count > 400);
        var mismatches = cases
            .Where(c => !c.Pieces.SequenceEqual(tokenizer.SplitForBpe(c.Text)))
            .Select(c => $"{JsonSerializer.Serialize(c.Text)}: expected {JsonSerializer.Serialize(c.Pieces)}, " +
                         $"got {JsonSerializer.Serialize(tokenizer.SplitForBpe(c.Text))}")
            .ToList();
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }

    /// <summary>End to end through the production GGUF dispatch, on a directory
    /// holding a GLM-5.x GGUF (e.g. GLM-5.3-Flash-UD-Q2_K_XL-*.gguf). Only the first
    /// shard of a split file is read: it carries the whole tokenizer metadata.</summary>
    [ModelFact("TS_TEST_MODEL_DIR", "glm-5")]
    public void Encode_FromGlm5Gguf_MatchesTheReferenceIds()
    {
        string path = TestGates.FindSmallestGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR")!, "glm-5")!;
        using var gguf = GgufFile.OpenWithoutSiblingShards(path);
        Assert.Equal("glm4", gguf.GetString("tokenizer.ggml.pre", null));
        var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var mismatches = Cases()
            .Where(c => !c.Ids.SequenceEqual(tokenizer.Encode(c.Text, addSpecial: false)))
            .Select(c => $"{JsonSerializer.Serialize(c.Text)}: expected [{string.Join(",", c.Ids)}], " +
                         $"got [{string.Join(",", tokenizer.Encode(c.Text, addSpecial: false))}]")
            .ToList();
        Assert.True(mismatches.Count == 0, string.Join("\n", mismatches));
    }
}
