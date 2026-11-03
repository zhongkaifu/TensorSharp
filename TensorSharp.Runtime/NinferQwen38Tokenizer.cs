// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime;

/// <summary>The known NInfer Qwen38 HF tokenizer profile. This does not enable model serving.</summary>
public sealed class NinferQwen38Tokenizer : ITokenizer, ISpecialTokenVocabulary
{
    private readonly string[] _vocab;
    private readonly Dictionary<string, int> _lookup;
    private readonly BpeTokenizer _encoder;
    private readonly int[] _eos;
    private readonly int _baseCount;
    private readonly IReadOnlyCollection<int> _special;
    private static readonly IReadOnlyDictionary<char, byte> ByteInverse = BuildByteInverse();
    internal const string RegexPattern = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    internal static readonly IReadOnlyList<(string Content, bool Special)> Added = Array.AsReadOnly(new[]
    {
        ("<|endoftext|>", true), ("<|im_start|>", true), ("<|im_end|>", true),
        ("<|object_ref_start|>", true), ("<|object_ref_end|>", true), ("<|box_start|>", true), ("<|box_end|>", true),
        ("<|quad_start|>", true), ("<|quad_end|>", true), ("<|vision_start|>", true), ("<|vision_end|>", true),
        ("<|vision_pad|>", true), ("<|image_pad|>", true), ("<|video_pad|>", true),
        ("<tool_call>", false), ("</tool_call>", false), ("<|fim_prefix|>", false), ("<|fim_middle|>", false),
        ("<|fim_suffix|>", false), ("<|fim_pad|>", false), ("<|repo_name|>", false), ("<|file_sep|>", false),
        ("<tool_response>", false), ("</tool_response>", false), ("<think>", false), ("</think>", false),
        ("<|audio_start|>", true), ("<|audio_end|>", true), ("<tts_pad>", true), ("<tts_text_bos>", true),
        ("<tts_text_eod>", true), ("<tts_text_bos_single>", true), ("<|audio_pad|>", true)
    });

    // ITokenizer's array properties return copies, so callers cannot change this profile.
    public string[] Vocab => (string[])_vocab.Clone();
    public int VocabSize => _vocab.Length;
    public int HeadVocabSize { get; }
    public int BosTokenId => -1; // HF tokenizer config has no BOS and add_bos_token=false.
    public int[] EosTokenIds => (int[])_eos.Clone();
    public IReadOnlyCollection<int> SpecialTokenIds => _special;
    public bool IsEos(int tokenId) => Array.IndexOf(_eos, tokenId) >= 0;
    public int LookupToken(string tokenStr) => _lookup.TryGetValue(tokenStr, out int id) ? id : -1;
    public List<int> Encode(string text, bool addSpecial = true)
    {
        ArgumentNullException.ThrowIfNull(text);
        // The legacy encoder splits all added tokens on raw text before NFC on plain fragments.
        // Its ISpecialTokenVocabulary is intentionally NOT exposed: user-defined != HF special.
        return _encoder.Encode(text, addSpecial);
    }
    public void AppendTokenBytes(int tokenId, List<byte> buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if ((uint)tokenId >= (uint)_vocab.Length) throw new ArgumentOutOfRangeException(nameof(tokenId), "No vocabulary entry for this ID (including padded head slots).");
        string token = _vocab[tokenId];
        if (tokenId >= _baseCount) { buffer.AddRange(Encoding.UTF8.GetBytes(token)); return; }
        foreach (char c in token) buffer.Add(ByteInverse[c]); // Includes U+0100 -> byte 0.
    }
    public string Decode(List<int> ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var bytes = new List<byte>();
        foreach (int id in ids) AppendTokenBytes(id, bytes);
        return Encoding.UTF8.GetString(bytes.ToArray()); // HF errors="replace", no whitespace cleanup.
    }
    private NinferQwen38Tokenizer(string[] vocab, string[] merges, int baseCount, int headCount, int[] eos)
    {
        _vocab = vocab; _baseCount = baseCount; HeadVocabSize = headCount; _eos = eos;
        _lookup = new Dictionary<string, int>(vocab.Length, StringComparer.Ordinal);
        for (int i = 0; i < vocab.Length; i++) _lookup.Add(vocab[i], i);
        var types = new int[vocab.Length]; Array.Fill(types, 1);
        for (int i = 0; i < Added.Count; i++) types[baseCount + i] = Added[i].Special ? 3 : 4;
        _special = Array.AsReadOnly(Enumerable.Range(0, Added.Count).Where(i => Added[i].Special).Select(i => baseCount + i).ToArray());
        _encoder = new BpeTokenizer(vocab, types, merges, -1, eos, false, false, "qwen35", normalizeNfc: true);
    }

    internal static NinferQwen38Tokenizer Parse(JsonElement root, JsonElement config, NinferQwen38GenerationMetadata generation, NinferFrontendProfile profile)
    {
        NinferFrontendJson.Exact(root, "version", "truncation", "padding", "added_tokens", "normalizer", "pre_tokenizer", "post_processor", "decoder", "model");
        NinferFrontendJson.Require(NinferFrontendJson.Text(root.GetProperty("version")) == "1.0" && root.GetProperty("truncation").ValueKind == JsonValueKind.Null && root.GetProperty("padding").ValueKind == JsonValueKind.Null, "Unsupported tokenizer version/truncation/padding.");
        var normalizer = root.GetProperty("normalizer"); NinferFrontendJson.Exact(normalizer, "type");
        NinferFrontendJson.Require(NinferFrontendJson.Text(normalizer.GetProperty("type")) == "NFC", "Only NFC is supported.");
        var pre = root.GetProperty("pre_tokenizer"); NinferFrontendJson.Exact(pre, "type", "pretokenizers");
        var passes = pre.GetProperty("pretokenizers"); NinferFrontendJson.Require(NinferFrontendJson.Text(pre.GetProperty("type")) == "Sequence" && passes.ValueKind == JsonValueKind.Array && passes.GetArrayLength() == 2, "Unsupported pre-tokenizer sequence.");
        var split = passes[0]; NinferFrontendJson.Exact(split, "type", "pattern", "behavior", "invert");
        var pattern = split.GetProperty("pattern"); NinferFrontendJson.Exact(pattern, "Regex");
        NinferFrontendJson.Require(NinferFrontendJson.Text(split.GetProperty("type")) == "Split" && NinferFrontendJson.Text(pattern.GetProperty("Regex")) == RegexPattern && NinferFrontendJson.Text(split.GetProperty("behavior")) == "Isolated" && !NinferFrontendJson.Bool(split.GetProperty("invert")), "Unsupported Qwen35 regex policy.");
        CheckByteLevel(passes[1]); CheckByteLevel(root.GetProperty("post_processor")); CheckByteLevel(root.GetProperty("decoder"));
        var model = root.GetProperty("model"); NinferFrontendJson.Exact(model, "type", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges", "vocab", "merges");
        NinferFrontendJson.Require(NinferFrontendJson.Text(model.GetProperty("type")) == "BPE" && model.GetProperty("dropout").ValueKind == JsonValueKind.Null && model.GetProperty("unk_token").ValueKind == JsonValueKind.Null && NinferFrontendJson.Text(model.GetProperty("continuing_subword_prefix")) == "" && NinferFrontendJson.Text(model.GetProperty("end_of_word_suffix")) == "" && !NinferFrontendJson.Bool(model.GetProperty("fuse_unk")) && !NinferFrontendJson.Bool(model.GetProperty("byte_fallback")) && !NinferFrontendJson.Bool(model.GetProperty("ignore_merges")), "Unsupported BPE behavior.");
        var vocabJson = model.GetProperty("vocab"); NinferFrontendJson.Require(vocabJson.ValueKind == JsonValueKind.Object && vocabJson.EnumerateObject().Count() == profile.BaseCount, "Base vocabulary count differs.");
        var vocab = new string[profile.BaseCount + Added.Count]; var occupied = new bool[profile.BaseCount];
        var baseLookup = new HashSet<string>(StringComparer.Ordinal);
        foreach (var row in vocabJson.EnumerateObject())
        {
            int id = NinferFrontendJson.Integer(row.Value);
            NinferFrontendJson.Require((uint)id < (uint)profile.BaseCount && !occupied[id] && row.Name.Length > 0 && row.Name.All(ByteInverse.ContainsKey), "Invalid/duplicate base ID or byte alphabet.");
            occupied[id] = true; vocab[id] = row.Name; baseLookup.Add(row.Name);
        }
        NinferFrontendJson.Require(occupied.All(x => x) && ByteInverse.Keys.All(c => baseLookup.Contains(c.ToString())), "Base vocabulary must cover the complete byte alphabet with contiguous IDs.");
        var mergesJson = model.GetProperty("merges"); NinferFrontendJson.Require(mergesJson.ValueKind == JsonValueKind.Array && mergesJson.GetArrayLength() == profile.MergeCount, "Merge count differs.");
        var merges = new string[profile.MergeCount]; var pairs = new HashSet<string>(StringComparer.Ordinal); var outputs = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < merges.Length; i++)
        {
            string merge = NinferFrontendJson.Text(mergesJson[i]); int space = merge.IndexOf(' ');
            NinferFrontendJson.Require(space > 0 && space < merge.Length - 1 && merge.IndexOf(' ', space + 1) < 0 && pairs.Add(merge), "Malformed/duplicate merge.");
            string left = merge[..space], right = merge[(space + 1)..];
            NinferFrontendJson.Require(baseLookup.Contains(left) && baseLookup.Contains(right) && baseLookup.Contains(left + right) && outputs.Add(left + right), "Merge input/output missing or duplicate output.");
            merges[i] = merge;
        }
        var added = root.GetProperty("added_tokens"); NinferFrontendJson.Require(added.ValueKind == JsonValueKind.Array && added.GetArrayLength() == Added.Count, "Added token count differs.");
        for (int i = 0; i < Added.Count; i++)
        {
            var token = added[i]; NinferFrontendJson.Exact(token, "id", "content", "single_word", "lstrip", "rstrip", "normalized", "special");
            NinferFrontendJson.Require(NinferFrontendJson.Integer(token.GetProperty("id")) == profile.BaseCount + i && NinferFrontendJson.Text(token.GetProperty("content")) == Added[i].Content && NinferFrontendJson.Bool(token.GetProperty("special")) == Added[i].Special, "Added token ID/content/special policy differs.");
            foreach (string k in new[] { "single_word", "lstrip", "rstrip", "normalized" }) NinferFrontendJson.Require(!NinferFrontendJson.Bool(token.GetProperty(k)), "Unsupported added-token matching policy.");
            NinferFrontendJson.Require(!baseLookup.Contains(Added[i].Content), "Added token overlaps base vocabulary."); vocab[profile.BaseCount + i] = Added[i].Content;
        }
        ValidateConfig(config, profile, generation);
        return new(vocab, merges, profile.BaseCount, profile.HeadCount, generation.EosTokenIds.ToArray());
    }
    private static void CheckByteLevel(JsonElement value)
    {
        NinferFrontendJson.Exact(value, "type", "add_prefix_space", "trim_offsets", "use_regex");
        NinferFrontendJson.Require(NinferFrontendJson.Text(value.GetProperty("type")) == "ByteLevel" && !NinferFrontendJson.Bool(value.GetProperty("add_prefix_space")) && !NinferFrontendJson.Bool(value.GetProperty("trim_offsets")) && !NinferFrontendJson.Bool(value.GetProperty("use_regex")), "Unsupported ByteLevel policy.");
    }
    private static void ValidateConfig(JsonElement c, NinferFrontendProfile p, NinferQwen38GenerationMetadata generation)
    {
        NinferFrontendJson.Exact(c, "add_prefix_space", "added_tokens_decoder", "additional_special_tokens", "bos_token", "chat_template", "clean_up_tokenization_spaces", "eos_token", "errors", "model_max_length", "pad_token", "split_special_tokens", "tokenizer_class", "unk_token", "add_bos_token", "pretokenize_regex", "extra_special_tokens");
        foreach (string k in new[] { "add_prefix_space", "clean_up_tokenization_spaces", "split_special_tokens", "add_bos_token" }) NinferFrontendJson.Require(!NinferFrontendJson.Bool(c.GetProperty(k)), "Unsupported tokenizer config behavior: " + k);
        NinferFrontendJson.Require(c.GetProperty("bos_token").ValueKind == JsonValueKind.Null && c.GetProperty("unk_token").ValueKind == JsonValueKind.Null && NinferFrontendJson.Text(c.GetProperty("eos_token")) == Added[2].Content && NinferFrontendJson.Text(c.GetProperty("pad_token")) == Added[0].Content && NinferFrontendJson.Text(c.GetProperty("errors")) == "replace" && NinferFrontendJson.Text(c.GetProperty("tokenizer_class")) == "Qwen2Tokenizer" && NinferFrontendJson.Integer(c.GetProperty("model_max_length")) == 262144 && NinferFrontendJson.Text(c.GetProperty("pretokenize_regex")) == RegexPattern, "Tokenizer config identity differs.");
        var decoder = c.GetProperty("added_tokens_decoder"); NinferFrontendJson.Require(decoder.ValueKind == JsonValueKind.Object && decoder.EnumerateObject().Count() == Added.Count, "Added decoder count differs.");
        for (int i = 0; i < Added.Count; i++)
        {
            string id = (p.BaseCount + i).ToString(System.Globalization.CultureInfo.InvariantCulture);
            NinferFrontendJson.Require(decoder.TryGetProperty(id, out var entry), "Missing added decoder ID."); NinferFrontendJson.Exact(entry, "content", "lstrip", "normalized", "rstrip", "single_word", "special");
            NinferFrontendJson.Require(NinferFrontendJson.Text(entry.GetProperty("content")) == Added[i].Content && NinferFrontendJson.Bool(entry.GetProperty("special")) == Added[i].Special, "Added decoder disagrees with tokenizer.");
            foreach (string k in new[] { "single_word", "lstrip", "rstrip", "normalized" }) NinferFrontendJson.Require(!NinferFrontendJson.Bool(entry.GetProperty(k)), "Unsupported added decoder policy.");
        }
        var additional = c.GetProperty("additional_special_tokens"); NinferFrontendJson.Require(additional.ValueKind == JsonValueKind.Array && additional.EnumerateArray().Select(NinferFrontendJson.Text).SequenceEqual(Added.Skip(1).Take(13).Select(x => x.Content)), "Additional special tokens differ.");
        var extras = c.GetProperty("extra_special_tokens"); NinferFrontendJson.Exact(extras, "audio_bos_token", "audio_eos_token", "audio_token", "image_token", "video_token", "vision_bos_token", "vision_eos_token");
        string[] keys = { "audio_bos_token", "audio_eos_token", "audio_token", "image_token", "video_token", "vision_bos_token", "vision_eos_token" }; int[] indices = { 26, 27, 32, 12, 13, 9, 10 };
        for (int i = 0; i < keys.Length; i++) NinferFrontendJson.Require(NinferFrontendJson.Text(extras.GetProperty(keys[i])) == Added[indices[i]].Content, "Extra special token differs.");
        NinferFrontendJson.Require(generation.BosTokenId == p.BaseCount && generation.PadTokenId == p.BaseCount && generation.EosTokenIds.SequenceEqual(new[] { p.BaseCount + 2, p.BaseCount }), "Generation token IDs disagree with vocabulary.");
    }
    private static IReadOnlyDictionary<char, byte> BuildByteInverse()
    {
        var bytes = Enumerable.Range(33, 94).Concat(Enumerable.Range(161, 12)).Concat(Enumerable.Range(174, 82)).ToList();
        var result = new Dictionary<char, byte>(); foreach (int b in bytes) result.Add((char)b, (byte)b);
        int extra = 0;
        for (int b = 0; b < 256; b++) if (!bytes.Contains(b)) result.Add((char)(256 + extra++), (byte)b);
        return new ReadOnlyDictionary<char, byte>(result);
    }
}

internal sealed record NinferFrontendProfile(int BaseCount, int MergeCount, int HeadCount)
{
    internal static readonly NinferFrontendProfile Actual = new(248044, 247587, 248320);
    internal static readonly NinferFrontendProfile TinyTests = new(258, 2, 320);
}
