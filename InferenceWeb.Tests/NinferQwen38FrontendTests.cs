using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// Self-contained generated alphabet/resources only. No ignored files, model or CUDA.
public sealed class NinferQwen38FrontendTests
{
    private const string Tokenizer = "frontend/tokenizer.json", Config = "frontend/tokenizer_config.json", Generation = "frontend/generation_config.json";
    private const string Template = "frontend/chat_template.jinja", Image = "frontend/preprocessor_config.json", Video = "frontend/video_preprocessor_config.json";
    private const int BaseCount = 258;
    // Independent GPT2 byte alphabet formula, not the production inverse helper.
    private static char ByteChar(int b) => b switch
    {
        <= 32 => (char)(256 + b),
        >= 127 and <= 160 => (char)(b + 162),
        173 => (char)323,
        _ => (char)b
    };
    private static JsonObject ByteLevel() => new() { ["type"] = "ByteLevel", ["add_prefix_space"] = false, ["trim_offsets"] = false, ["use_regex"] = false };
    private static Dictionary<string, byte[]> Fixture()
    {
        var vocab = new JsonObject(); for (int b = 0; b < 256; b++) vocab.Add(ByteChar(b).ToString(), b);
        vocab.Add("hi", 256); vocab.Add(ByteChar(32) + "hi", 257);
        var added = new JsonArray(); var decoder = new JsonObject();
        for (int i = 0; i < NinferQwen38Tokenizer.Added.Count; i++)
        {
            var spec = NinferQwen38Tokenizer.Added[i];
            var entry = new JsonObject { ["content"] = spec.Content, ["single_word"] = false, ["lstrip"] = false, ["rstrip"] = false, ["normalized"] = false, ["special"] = spec.Special };
            decoder.Add((BaseCount + i).ToString(), entry.DeepClone()); entry.Add("id", BaseCount + i); added.Add(entry);
        }
        var tokenizer = new JsonObject
        {
            ["version"] = "1.0", ["truncation"] = null, ["padding"] = null, ["added_tokens"] = added,
            ["normalizer"] = new JsonObject { ["type"] = "NFC" },
            ["pre_tokenizer"] = new JsonObject { ["type"] = "Sequence", ["pretokenizers"] = new JsonArray(new JsonObject { ["type"] = "Split", ["pattern"] = new JsonObject { ["Regex"] = NinferQwen38Tokenizer.RegexPattern }, ["behavior"] = "Isolated", ["invert"] = false }, ByteLevel()) },
            ["post_processor"] = ByteLevel(), ["decoder"] = ByteLevel(),
            ["model"] = new JsonObject { ["type"] = "BPE", ["dropout"] = null, ["unk_token"] = null, ["continuing_subword_prefix"] = "", ["end_of_word_suffix"] = "", ["fuse_unk"] = false, ["byte_fallback"] = false, ["ignore_merges"] = false, ["vocab"] = vocab, ["merges"] = new JsonArray("h i", ByteChar(32) + " hi") }
        };
        const string template = "{{ messages }}\n原始模板\n";
        var config = new JsonObject
        {
            ["add_prefix_space"] = false, ["added_tokens_decoder"] = decoder,
            ["additional_special_tokens"] = new JsonArray(NinferQwen38Tokenizer.Added.Skip(1).Take(13).Select(x => JsonValue.Create(x.Content)).ToArray()),
            ["bos_token"] = null, ["chat_template"] = template, ["clean_up_tokenization_spaces"] = false, ["eos_token"] = "<|im_end|>", ["errors"] = "replace", ["model_max_length"] = 262144,
            ["pad_token"] = "<|endoftext|>", ["split_special_tokens"] = false, ["tokenizer_class"] = "Qwen2Tokenizer", ["unk_token"] = null, ["add_bos_token"] = false, ["pretokenize_regex"] = NinferQwen38Tokenizer.RegexPattern,
            ["extra_special_tokens"] = new JsonObject { ["audio_bos_token"] = "<|audio_start|>", ["audio_eos_token"] = "<|audio_end|>", ["audio_token"] = "<|audio_pad|>", ["image_token"] = "<|image_pad|>", ["video_token"] = "<|video_pad|>", ["vision_bos_token"] = "<|vision_start|>", ["vision_eos_token"] = "<|vision_end|>" }
        };
        var generation = new JsonObject { ["bos_token_id"] = BaseCount, ["do_sample"] = true, ["eos_token_id"] = new JsonArray(BaseCount + 2, BaseCount), ["pad_token_id"] = BaseCount, ["temperature"] = 1.0, ["top_k"] = 20, ["top_p"] = 0.95 };
        static JsonObject Processor(bool video) => new()
        {
            ["size"] = new JsonObject { ["longest_edge"] = video ? 25165824 : 16777216, ["shortest_edge"] = video ? 4096 : 65536 },
            ["patch_size"] = 16, ["temporal_patch_size"] = 2, ["merge_size"] = 2,
            ["image_mean"] = new JsonArray(0.5, 0.5, 0.5), ["image_std"] = new JsonArray(0.5, 0.5, 0.5), ["processor_class"] = "Qwen3VLProcessor",
            [video ? "video_processor_type" : "image_processor_type"] = video ? "Qwen3VLVideoProcessor" : "Qwen2VLImageProcessorFast"
        };
        return new Dictionary<string, byte[]>(StringComparer.Ordinal)
        {
            [Tokenizer] = Encoding.UTF8.GetBytes(tokenizer.ToJsonString()), [Config] = Encoding.UTF8.GetBytes(config.ToJsonString()),
            [Generation] = Encoding.UTF8.GetBytes(generation.ToJsonString()), [Template] = Encoding.UTF8.GetBytes(template),
            [Image] = Encoding.UTF8.GetBytes(Processor(false).ToJsonString()), [Video] = Encoding.UTF8.GetBytes(Processor(true).ToJsonString())
        };
    }
    private static void Mutate(Dictionary<string, byte[]> bytes, string name, Action<JsonObject> action)
    {
        var root = JsonNode.Parse(bytes[name])!.AsObject(); action(root); bytes[name] = Encoding.UTF8.GetBytes(root.ToJsonString());
    }

    [Fact]
    public void FullByteInversePreservesEveryByteIncludingZeroAndReplacementDecoding()
    {
        var t = NinferQwen38Frontend.ParseTinyForTests(Fixture()).Tokenizer;
        var buffer = new List<byte>();
        for (int b = 0; b < 256; b++) t.AppendTokenBytes(b, buffer);
        Assert.Equal(Enumerable.Range(0, 256).Select(x => (byte)x), buffer);
        Assert.Equal("\0", t.Decode(new List<int> { 0 }));
        Assert.Equal("�", t.Decode(new List<int> { 255 }));
        Assert.Equal(new[] { 0, 1, 2 }, t.Encode("\0\x01\x02"));
    }

    [Fact]
    public void AllAddedTokensAreAtomicButOnlyHfSpecialTrueIdsAreSpecial()
    {
        var t = NinferQwen38Frontend.ParseTinyForTests(Fixture()).Tokenizer;
        Assert.Equal(33, NinferQwen38Tokenizer.Added.Count); Assert.Equal(21, t.SpecialTokenIds.Count);
        for (int i = 0; i < 33; i++)
        {
            var spec = NinferQwen38Tokenizer.Added[i]; int id = BaseCount + i;
            Assert.Equal(new[] { id }, t.Encode(spec.Content, addSpecial: false));
            Assert.Equal(spec.Content, t.Decode(new List<int> { id }));
            Assert.Equal(spec.Special, t.SpecialTokenIds.Contains(id));
        }
        Assert.DoesNotContain(t.LookupToken("<tool_call>"), ((ISpecialTokenVocabulary)t).SpecialTokenIds);
    }

    [Fact]
    public void NfcAndBpeMergesRespectRawAddedFragmentBoundariesAndNoAutomaticBosEos()
    {
        var t = NinferQwen38Frontend.ParseTinyForTests(Fixture()).Tokenizer;
        Assert.Equal(new[] { 256 }, t.Encode("hi")); Assert.Equal(new[] { 257 }, t.Encode(" hi"));
        Assert.Equal(new[] { 104, 256 }, t.Encode("hhi"));
        int added = t.LookupToken("<think>");
        Assert.Equal(new[] { 195, 169, added, 195, 169 }, t.Encode("e\u0301<think>e\u0301"));
        Assert.Equal(t.Encode("é"), t.Encode("e\u0301")); Assert.Equal("é", t.Decode(t.Encode("e\u0301")));
        Assert.Equal(t.Encode("hi", false), t.Encode("hi", true)); Assert.Empty(t.Encode(""));
        Assert.Equal(-1, t.BosTokenId); Assert.Equal(new[] { BaseCount + 2, BaseCount }, t.EosTokenIds);
    }

    [Fact]
    public void VocabularyAndPaddedHeadHaveDifferentDomainsWithoutFakeTokensOrMasking()
    {
        var f = NinferQwen38Frontend.ParseTinyForTests(Fixture()); var t = f.Tokenizer;
        Assert.Equal(291, t.VocabSize); Assert.Equal(320, t.HeadVocabSize);
        Assert.Throws<ArgumentOutOfRangeException>(() => t.AppendTokenBytes(291, new List<byte>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => t.Decode(new List<int> { 319 }));
        Assert.Equal(-1, t.LookupToken("fake-padding-319")); Assert.False(t.IsEos(319));
        Assert.False(f.InferenceSupported); Assert.False(f.SamplingSettingsApplied); Assert.False(f.ChatTemplateRenderingImplemented); Assert.False(f.VisionProcessingImplemented);
    }

    [Fact]
    public void MetadataAndRawResourcesAreImmutablePreservedAndDoNotApplySampling()
    {
        var bytes = Fixture(); var f = NinferQwen38Frontend.ParseTinyForTests(bytes);
        Assert.Equal(1.0, f.Generation.Temperature); Assert.Equal(20, f.Generation.TopK); Assert.Equal(0.95, f.Generation.TopP); Assert.True(f.Generation.DoSample);
        Assert.Equal(BaseCount, f.Generation.BosTokenId); Assert.Equal(-1, f.Tokenizer.BosTokenId);
        Assert.Equal(262144, f.MaximumContextTokens); Assert.Equal(6, f.Resources.Count);
        foreach (var pair in bytes)
        {
            var resource = f.Resources[pair.Key]; Assert.Equal(pair.Value, resource.CopyBytes());
            Assert.Equal(Convert.ToHexString(SHA256.HashData(pair.Value)).ToLowerInvariant(), resource.Sha256);
            resource.CopyBytes()[0] ^= 1; Assert.Equal(pair.Value, resource.CopyBytes());
        }
        Assert.Equal(Encoding.UTF8.GetString(bytes[Template]), f.ChatTemplate);
        Assert.Throws<NotSupportedException>(() => ((IList<int>)f.Generation.EosTokenIds)[0] = 0);
        Assert.Throws<NotSupportedException>(() => ((IList<double>)f.ImageProcessor.Mean)[0] = 0);
        var vocab = f.Tokenizer.Vocab; vocab[256] = "corrupt"; var eos = f.Tokenizer.EosTokenIds; eos[0] = 0;
        Assert.Equal(new[] { 256 }, f.Tokenizer.Encode("hi")); Assert.True(f.Tokenizer.IsEos(BaseCount + 2));
        bytes[Template][0] ^= 1; Assert.StartsWith("{{", f.ChatTemplate);
    }

    [Theory]
    [InlineData("normalizer")][InlineData("regex")][InlineData("bytelevel")][InlineData("dropout")][InlineData("unknown")]
    [InlineData("added_normalized")][InlineData("added_special")][InlineData("added_id")][InlineData("decoder_config")][InlineData("bos")]
    [InlineData("eos")][InlineData("template")][InlineData("processor")]
    public void UnsupportedBehaviorAndDisagreeingConfigsAreRefused(string mutation)
    {
        var b = Fixture();
        switch (mutation)
        {
            case "normalizer": Mutate(b, Tokenizer, x => x["normalizer"]!["type"] = "NFKC"); break;
            case "regex": Mutate(b, Tokenizer, x => x["pre_tokenizer"]!["pretokenizers"]![0]!["pattern"]!["Regex"] = ".*"); break;
            case "bytelevel": Mutate(b, Tokenizer, x => x["decoder"]!["use_regex"] = true); break;
            case "dropout": Mutate(b, Tokenizer, x => x["model"]!["dropout"] = 0.1); break;
            case "unknown": Mutate(b, Tokenizer, x => x["new_behavior"] = false); break;
            case "added_normalized": Mutate(b, Tokenizer, x => x["added_tokens"]![14]!["normalized"] = true); break;
            case "added_special": Mutate(b, Tokenizer, x => x["added_tokens"]![14]!["special"] = true); break;
            case "added_id": Mutate(b, Tokenizer, x => x["added_tokens"]![1]!["id"] = BaseCount); break;
            case "decoder_config": Mutate(b, Config, x => x["added_tokens_decoder"]![BaseCount.ToString()]!["special"] = false); break;
            case "bos": Mutate(b, Config, x => x["add_bos_token"] = true); break;
            case "eos": Mutate(b, Generation, x => x["eos_token_id"] = new JsonArray(BaseCount, BaseCount + 2)); break;
            case "template": b[Template] = Encoding.UTF8.GetBytes("changed\n"); break;
            case "processor": Mutate(b, Image, x => x["patch_size"] = 14); break;
        }
        Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
    }

    [Theory]
    [InlineData("id_duplicate")][InlineData("id_outside")][InlineData("id_boolean")][InlineData("alphabet")]
    [InlineData("merge_duplicate")][InlineData("merge_unknown")][InlineData("merge_array")]
    public void CorruptVocabularyAndMergeGeometryAreRefused(string mutation)
    {
        var b = Fixture();
        Mutate(b, Tokenizer, x =>
        {
            var model = x["model"]!;
            switch (mutation)
            {
                case "id_duplicate": model["vocab"]!["hi"] = 0; break;
                case "id_outside": model["vocab"]!["hi"] = 1000; break;
                case "id_boolean": model["vocab"]!["hi"] = true; break;
                case "alphabet": model["vocab"]!.AsObject().Remove(ByteChar(0).ToString()); model["vocab"]!["▁"] = 0; break;
                case "merge_duplicate": model["merges"]![1] = "h i"; break;
                case "merge_unknown": model["merges"]![0] = "h not-in-vocab"; break;
                case "merge_array": model["merges"]![0] = new JsonArray("h", "i"); break;
            }
        });
        Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
    }

    [Fact]
    public void DuplicateJsonAndInvalidUtf8OrMissingResourcesAreNotSilentlyAccepted()
    {
        var b = Fixture(); string json = Encoding.UTF8.GetString(b[Tokenizer]);
        b[Tokenizer] = Encoding.UTF8.GetBytes("{\"version\":\"1.0\"," + json[1..]);
        Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
        b = Fixture(); b[Template] = new byte[] { 255 }; Assert.Throws<DecoderFallbackException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
        b = Fixture(); b.Remove(Generation); Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
        b = Fixture(); b["foreign/resource"] = new byte[] { 1 }; Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
        b = Fixture(); b[Template] = new byte[(64 << 10) + 1]; Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.ParseTinyForTests(b));
    }

    [Fact]
    public void ExplicitSamplingMetadataIsKeptRatherThanReplacedWithBenchmarkDefaults()
    {
        var b = Fixture(); Mutate(b, Generation, x => { x["do_sample"] = false; x["temperature"] = 0.25; x["top_k"] = 7; x["top_p"] = 0.8; });
        var f = NinferQwen38Frontend.ParseTinyForTests(b);
        Assert.False(f.Generation.DoSample); Assert.Equal(0.25, f.Generation.Temperature); Assert.Equal(7, f.Generation.TopK); Assert.Equal(0.8, f.Generation.TopP); Assert.False(f.SamplingSettingsApplied);
    }

    [Fact]
    public void PublicReaderPathRefusesTinyOrUnknownInventoryBeforeReadingResources()
    {
        Span<byte> len = stackalloc byte[8];
        foreach (string family in new[] { "tiny", "qwen3.8-27b" })
        {
            string dir = Path.Combine(Path.GetTempPath(), "ninfer-front-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(dir);
            try
            {
                string path = Path.Combine(dir, "tiny.ninfer");
                byte[] metadata = JsonSerializer.SerializeToUtf8Bytes(new { identity = new { model_id = family, weights_id = "nvfp4full" }, objects = new[] { new { name = Tokenizer, kind = "resource", encoding = "raw-bytes-v1", offset = 0, bytes = 1 } } });
                using (var file = File.Create(path)) { file.Write("NINFER\0\x02"u8); BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)metadata.Length); file.Write(len); file.Write(metadata); file.SetLength((16L + metadata.Length + 4095) / 4096 * 4096 + 1); }
                using var reader = new NinferArtifactReader(path); long before = reader.TotalBytesRead;
                Assert.Throws<InvalidDataException>(() => NinferQwen38Frontend.FromReader(reader)); Assert.Equal(before, reader.TotalBytesRead);
                reader.Dispose(); Assert.Throws<ObjectDisposedException>(() => NinferQwen38Frontend.FromReader(reader));
            }
            finally { Directory.Delete(dir, true); }
        }
    }
}
