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
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Whisper
{
    /// <summary>Hyperparameters and special tokens of a Whisper GGUF (eng/audio-models/whisper_to_gguf.py),
    /// read from the Hugging Face config.json and generation_config.json it carries.</summary>
    public sealed class WhisperConfig
    {
        public int DModel { get; private init; }
        public int EncoderLayers { get; private init; }
        public int EncoderHeads { get; private init; }
        public int DecoderLayers { get; private init; }
        public int DecoderHeads { get; private init; }
        public int EncoderFfn { get; private init; }
        public int DecoderFfn { get; private init; }
        public int NumMelBins { get; private init; }
        public int VocabSize { get; private init; }
        public int MaxSourcePositions { get; private init; }
        public int MaxTargetPositions { get; private init; }
        public int NFft { get; private init; }
        public int HopLength { get; private init; }
        public int SampleRate { get; private init; }
        public int ChunkSeconds { get; private init; }

        /// <summary>Samples in one 30-second window (480,000).</summary>
        public int ChunkSamples => ChunkSeconds * SampleRate;
        /// <summary>Mel frames in one window (3,000).</summary>
        public int ChunkFrames => ChunkSamples / HopLength;
        /// <summary>Seconds per timestamp token (0.02: two mel frames per encoder position).</summary>
        public double TimePrecision => 2.0 * HopLength / SampleRate;

        public int EndOfText { get; private init; }
        public int StartOfTranscript { get; private init; }
        public int Translate { get; private init; }
        public int Transcribe { get; private init; }
        public int StartOfLm { get; private init; }
        public int StartOfPrev { get; private init; }
        public int NoSpeech { get; private init; }
        public int NoTimestamps { get; private init; }
        public int TimestampBegin { get; private init; }
        public bool IsMultilingual { get; private init; }
        public int MaxInitialTimestampIndex { get; private init; } = 50;
        public IReadOnlyDictionary<string, int> LanguageTokens { get; private init; }
        public int[] SuppressTokens { get; private init; }
        public int[] BeginSuppressTokens { get; private init; }
        public (int Layer, int Head)[] AlignmentHeads { get; private init; }
        public int MedianFilterWidth { get; private init; } = 7;

        public static WhisperConfig Read(GgufFile gguf, string[] vocab)
        {
            string cfgJson = gguf.GetString("whisper.config")
                ?? throw new InvalidDataException("not a TensorSharp Whisper GGUF: whisper.config is missing " +
                                                  "(convert the checkpoint with eng/audio-models/whisper_to_gguf.py).");
            string genJson = gguf.GetString("whisper.generation_config") ?? "{}";
            using var cfg = JsonDocument.Parse(cfgJson);
            using var gen = JsonDocument.Parse(genJson);
            var c = cfg.RootElement;
            var g = gen.RootElement;

            int Int(JsonElement e, string key, int fallback = 0) =>
                e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

            int Token(string text)
            {
                int i = Array.IndexOf(vocab, text);
                return i;
            }

            var languages = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            if (g.TryGetProperty("lang_to_id", out var langs) && langs.ValueKind == JsonValueKind.Object)
            {
                foreach (var p in langs.EnumerateObject())
                {
                    string code = p.Name.Trim('<', '>', '|');
                    languages[code] = p.Value.GetInt32();
                }
            }

            int[] Ints(JsonElement e, string key)
            {
                if (!e.TryGetProperty(key, out var v) || v.ValueKind != JsonValueKind.Array) return Array.Empty<int>();
                return v.EnumerateArray().Where(x => x.ValueKind == JsonValueKind.Number).Select(x => x.GetInt32()).ToArray();
            }

            var heads = new List<(int, int)>();
            if (g.TryGetProperty("alignment_heads", out var ah) && ah.ValueKind == JsonValueKind.Array)
                foreach (var pair in ah.EnumerateArray())
                {
                    var two = pair.EnumerateArray().Select(x => x.GetInt32()).ToArray();
                    if (two.Length == 2) heads.Add((two[0], two[1]));
                }

            int noSpeech = Token("<|nospeech|>");
            if (noSpeech < 0) noSpeech = Token("<|nocaptions|>");
            var config = new WhisperConfig
            {
                DModel = Int(c, "d_model"),
                EncoderLayers = Int(c, "encoder_layers"),
                EncoderHeads = Int(c, "encoder_attention_heads"),
                DecoderLayers = Int(c, "decoder_layers"),
                DecoderHeads = Int(c, "decoder_attention_heads"),
                EncoderFfn = Int(c, "encoder_ffn_dim"),
                DecoderFfn = Int(c, "decoder_ffn_dim"),
                NumMelBins = Int(c, "num_mel_bins", 80),
                VocabSize = Int(c, "vocab_size"),
                MaxSourcePositions = Int(c, "max_source_positions", 1500),
                MaxTargetPositions = Int(c, "max_target_positions", 448),
                MedianFilterWidth = Int(c, "median_filter_width", 7),
                NFft = (int)gguf.GetUint32("whisper.n_fft", 400),
                HopLength = (int)gguf.GetUint32("whisper.hop_length", 160),
                SampleRate = (int)gguf.GetUint32("whisper.sample_rate", 16000),
                ChunkSeconds = (int)gguf.GetUint32("whisper.chunk_length", 30),
                EndOfText = Token("<|endoftext|>"),
                StartOfTranscript = Token("<|startoftranscript|>"),
                Translate = Token("<|translate|>"),
                Transcribe = Token("<|transcribe|>"),
                StartOfLm = Token("<|startoflm|>"),
                StartOfPrev = Token("<|startofprev|>"),
                NoSpeech = noSpeech,
                NoTimestamps = Token("<|notimestamps|>"),
                TimestampBegin = Token("<|0.00|>"),
                IsMultilingual = languages.Count > 0,
                MaxInitialTimestampIndex = Int(g, "max_initial_timestamp_index", 50),
                LanguageTokens = languages,
                SuppressTokens = Ints(g, "suppress_tokens"),
                BeginSuppressTokens = Ints(g, "begin_suppress_tokens"),
                AlignmentHeads = heads.ToArray(),
            };
            if (config.DModel <= 0 || config.EncoderLayers <= 0 || config.DecoderLayers <= 0)
                throw new InvalidDataException("whisper.config lacks d_model / encoder_layers / decoder_layers.");
            if (config.EndOfText < 0 || config.StartOfTranscript < 0 || config.TimestampBegin < 0 || config.NoTimestamps < 0)
                throw new InvalidDataException("the Whisper vocabulary lacks <|endoftext|>, <|startoftranscript|>, " +
                                               "<|notimestamps|> or the <|0.00|> timestamp token.");
            return config;
        }

        public override string ToString() =>
            $"Whisper: d_model={DModel} encoder={EncoderLayers}x{EncoderHeads}h decoder={DecoderLayers}x{DecoderHeads}h " +
            $"mels={NumMelBins} vocab={VocabSize} languages={LanguageTokens.Count}";
    }

    /// <summary>Whisper's byte-level BPE: decoding that keeps or drops the special tokens, and encoding for prompts.</summary>
    public sealed class WhisperTokenizer
    {
        private readonly string[] _vocab;
        private readonly BpeTokenizer _bpe;
        private readonly WhisperConfig _cfg;
        private static readonly Dictionary<char, byte> ByteDecoder = BuildByteDecoder();

        public WhisperTokenizer(string[] vocab, int[] types, string[] merges, WhisperConfig cfg)
        {
            _vocab = vocab;
            _cfg = cfg;
            _bpe = new BpeTokenizer(vocab, types, merges, cfg.EndOfText, new[] { cfg.EndOfText },
                addBos: false, addEos: false, preTokenizerType: "whisper");
        }

        public string TokenText(int id) => id >= 0 && id < _vocab.Length ? _vocab[id] : string.Empty;

        /// <summary>Text tokens only (ids below &lt;|endoftext|&gt;), as tokenizer.decode does.</summary>
        public string Decode(IEnumerable<int> ids) => DecodeCore(ids, keepSpecial: false);

        /// <summary>Every token, special ones spelled out (&lt;|1.24|&gt;, &lt;|endoftext|&gt;).</summary>
        public string DecodeWithTimestamps(IEnumerable<int> ids) => DecodeCore(ids, keepSpecial: true);

        private string DecodeCore(IEnumerable<int> ids, bool keepSpecial)
        {
            var sb = new StringBuilder();
            var bytes = new List<byte>();
            void Flush()
            {
                if (bytes.Count > 0) { sb.Append(Encoding.UTF8.GetString(bytes.ToArray())); bytes.Clear(); }
            }
            foreach (int id in ids)
            {
                if (id < 0 || id >= _vocab.Length) continue;
                if (id >= _cfg.EndOfText)
                {
                    if (!keepSpecial) continue;
                    Flush();
                    sb.Append(_vocab[id]);
                    continue;
                }
                foreach (char ch in _vocab[id])
                {
                    if (ByteDecoder.TryGetValue(ch, out byte b)) bytes.Add(b);
                    else bytes.AddRange(Encoding.UTF8.GetBytes(ch.ToString()));
                }
            }
            Flush();
            return sb.ToString();
        }

        /// <summary>Text to ids, special tokens not recognised (a prompt is plain text).</summary>
        public List<int> Encode(string text) => _bpe.Encode(text, addSpecial: false);

        /// <summary>GPT-2's bytes_to_unicode, inverted.</summary>
        private static Dictionary<char, byte> BuildByteDecoder()
        {
            var bs = new List<int>();
            for (int b = '!'; b <= '~'; b++) bs.Add(b);
            for (int b = 0xA1; b <= 0xAC; b++) bs.Add(b);
            for (int b = 0xAE; b <= 0xFF; b++) bs.Add(b);
            var cs = new List<int>(bs);
            int n = 0;
            for (int b = 0; b < 256; b++)
            {
                if (!bs.Contains(b)) { bs.Add(b); cs.Add(256 + n); n++; }
            }
            var map = new Dictionary<char, byte>();
            for (int i = 0; i < bs.Count; i++) map[(char)cs[i]] = (byte)bs[i];
            return map;
        }
    }
}
