using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Ltx25TokenizerAssetTheoryAttribute : TheoryAttribute
{
    public Ltx25TokenizerAssetTheoryAttribute()
    {
        if (!File.Exists(Environment.GetEnvironmentVariable("TS_TEST_LTX25_TOKENIZER_ENCODER")))
            Skip = "Set TS_TEST_LTX25_TOKENIZER_ENCODER to the pinned Gemma4 LTX encoder. Only its 32 MB tokenizer asset is read; no model tensors or GPU are used.";
    }
}

public sealed class Ltx25MultilingualTokenizerTests
{
    private static JsonElement Fixture()
    {
        using var stream = typeof(Ltx25MultilingualTokenizerTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Ltx25.tokenizer-multilingual.json")!;
        using var document = JsonDocument.Parse(stream);
        return document.RootElement.Clone();
    }

    private static readonly Lazy<Ltx25PromptTokenizer> Tokenizer = new(() =>
    {
        using var file = new SafetensorsFile(Environment.GetEnvironmentVariable("TS_TEST_LTX25_TOKENIZER_ENCODER")!);
        byte[] bytes = file.ReadRawBytes("tokenizer_json");
        Assert.Equal(Fixture().GetProperty("tokenizer_json_sha256").GetString(), Convert.ToHexStringLower(SHA256.HashData(bytes)));
        Assert.InRange(bytes.Length, 1, 64 * 1024 * 1024);
        return new Ltx25PromptTokenizer(bytes);
    });

    public static IEnumerable<object[]> Cases() => Fixture().GetProperty("cases").EnumerateArray()
        .Select(test => new object[] { test.GetProperty("name").GetString()! });

    [Ltx25TokenizerAssetTheory]
    [MemberData(nameof(Cases))]
    public void ActualTokenizerIdsNormalizationTruncationAndPaddingMatchUnchangedOfficial(string name)
    {
        JsonElement fixture = Fixture();
        Assert.Equal("be6d6d0a69d11a0a399c462ed92606692e5f8697", fixture.GetProperty("source_revision").GetString());
        Assert.Equal("5.10.1", fixture.GetProperty("transformers_version").GetString());
        Assert.Equal(0, fixture.GetProperty("model_tensor_bytes_read").GetInt32());
        var test = fixture.GetProperty("cases").EnumerateArray().Single(value => value.GetProperty("name").GetString() == name);
        string prompt = test.GetProperty("prompt").GetString()!;
        Assert.Equal(test.GetProperty("trimmed_prompt").GetString(), prompt.Trim());
        int[] actual = Tokenizer.Value.Encode(prompt);
        int[] expected = test.GetProperty("ids").EnumerateArray().Select(value => value.GetInt32()).ToArray();
        Assert.Equal(expected, actual);
        Assert.Equal(fixture.GetProperty("maximum_length").GetInt32(), Tokenizer.Value.MaximumLength);
        Assert.Equal(Math.Min(test.GetProperty("untruncated_tokens").GetInt32(), Tokenizer.Value.MaximumLength), actual.Length);
        Assert.Equal(2, actual[0]);
        if (test.GetProperty("truncated").GetBoolean()) Assert.Equal(Tokenizer.Value.MaximumLength, actual.Length);

        // Native Gemma executes only valid tokens but keeps these original
        // left-padded positions; connector preparation moves valid rows ahead
        // of padding before replacing the remaining rows by registers.
        Assert.Equal("left", fixture.GetProperty("padding_side").GetString());
        int padding = Tokenizer.Value.MaximumLength - actual.Length;
        int[] padded = Enumerable.Repeat(fixture.GetProperty("pad_token_id").GetInt32(), padding).Concat(actual).ToArray();
        int[] mask = Enumerable.Repeat(0, padding).Concat(Enumerable.Repeat(1, actual.Length)).ToArray();
        Assert.Equal(test.GetProperty("padded_ids").EnumerateArray().Select(value => value.GetInt32()), padded);
        Assert.Equal(test.GetProperty("attention_mask").EnumerateArray().Select(value => value.GetInt32()), mask);
        Assert.Equal(test.GetProperty("valid_position_ids").EnumerateArray().Select(value => value.GetInt32()), Enumerable.Range(padding, actual.Length));
        var order = Ltx25HiddenStateFeatures.CreateRightPadOrder(mask);
        Assert.Equal(actual, order.SourceTokenIndices.Take(actual.Length).Select(index => padded[index]));
        Assert.Equal(Enumerable.Repeat(1, actual.Length).Concat(Enumerable.Repeat(0, padding)), order.AttentionMask);
    }
}
