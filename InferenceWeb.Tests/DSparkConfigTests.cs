using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using Xunit;
using TensorSharp;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// The DSpark parts of <see cref="DFlashConfig.FromGguf"/>: Markov-head
    /// detection (rank from the markov_w1.weight shape, since llama.cpp's export
    /// carries no dflash.markov_rank key), attention-sink detection, the
    /// missing-key defaults (sample_from_anchor = true like llama.cpp,
    /// sliding_window = 1024 for DSpark files, all-sliding pattern) and the
    /// draft-count consequence (a Markov drafter keeps the whole block width).
    /// </summary>
    public class DSparkConfigTests
    {
        private sealed class TinyGguf
        {
            private readonly List<(string key, int type, object value)> _kv = new();
            private readonly List<(string name, int ggmlType, byte[] payload)> _tensors = new();

            public void AddString(string k, string v) => _kv.Add((k, 8, v));
            public void AddUint32(string k, uint v) => _kv.Add((k, 4, v));
            public void AddFloat32(string k, float v) => _kv.Add((k, 6, v));
            public void AddBool(string k, bool v) => _kv.Add((k, 7, v));
            public void AddInt32Array(string k, int[] v) => _kv.Add((k, 9, v));
            public void AddBoolArray(string k, bool[] v) => _kv.Add((k, 9, v));

            public void AddTensor(string name, int ggmlType, int count)
                => _tensors.Add((name, ggmlType, new byte[count * 2]));

            public string Write()
            {
                string path = Path.Combine(Path.GetTempPath(), "dspark-cfg-" + Guid.NewGuid().ToString("N") + ".gguf");
                using (var f = File.Create(path))
                {
                    void Str(string s)
                    {
                        byte[] b = Encoding.UTF8.GetBytes(s);
                        f.Write(BitConverter.GetBytes((long)b.Length));
                        f.Write(b);
                    }
                    f.Write(Encoding.ASCII.GetBytes("GGUF"));
                    f.Write(BitConverter.GetBytes(3u));
                    f.Write(BitConverter.GetBytes((long)_tensors.Count));
                    f.Write(BitConverter.GetBytes((long)_kv.Count));
                    foreach (var (key, type, value) in _kv)
                    {
                        Str(key);
                        f.Write(BitConverter.GetBytes((uint)type));
                        switch (type)
                        {
                            case 8: Str((string)value); break;
                            case 4: f.Write(BitConverter.GetBytes((uint)value)); break;
                            case 6: f.Write(BitConverter.GetBytes((float)value)); break;
                            case 7: f.WriteByte((byte)((bool)value ? 1 : 0)); break;
                            case 9:
                                if (value is bool[] bArr)
                                {
                                    f.Write(BitConverter.GetBytes(7u)); // BOOL elements
                                    f.Write(BitConverter.GetBytes((long)bArr.Length));
                                    foreach (bool v in bArr)
                                        f.WriteByte((byte)(v ? 1 : 0));
                                }
                                else
                                {
                                    var arr = (int[])value;
                                    f.Write(BitConverter.GetBytes(5u)); // INT32 elements
                                    f.Write(BitConverter.GetBytes((long)arr.Length));
                                    foreach (int v in arr)
                                        f.Write(BitConverter.GetBytes(v));
                                }
                                break;
                        }
                    }
                    long offset = 0;
                    var infos = new MemoryStream();
                    foreach (var (name, ggmlType, payload) in _tensors)
                    {
                        byte[] nameBytes = Encoding.UTF8.GetBytes(name);
                        infos.Write(BitConverter.GetBytes((long)nameBytes.Length));
                        infos.Write(nameBytes);
                        infos.Write(BitConverter.GetBytes(2u));  // n_dims
                        infos.Write(BitConverter.GetBytes((long)512));   // ne0
                        infos.Write(BitConverter.GetBytes((long)16));    // ne1
                        infos.Write(BitConverter.GetBytes((uint)ggmlType));
                        infos.Write(BitConverter.GetBytes(offset));
                        offset += (payload.Length + 31) & ~31;
                    }
                    byte[] infoBytes = infos.ToArray();
                    int pad = (32 - ((int)f.Position + infoBytes.Length) % 32) % 32;
                    f.Write(infoBytes);
                    f.Write(new byte[pad]);
                    foreach (var (_, _, payload) in _tensors)
                    {
                        f.Write(payload);
                        f.Write(new byte[(32 - payload.Length % 32) % 32]);
                    }
                }
                return path;
            }
        }

        private static TinyGguf BaseDraft(bool withMarkov, bool withSinks, bool withSwa, bool withSampleAnchor)
        {
            var g = new TinyGguf();
            g.AddString("general.architecture", "dflash");
            g.AddUint32("dflash.block_count", 2);
            g.AddUint32("dflash.embedding_length", 16);
            g.AddUint32("dflash.feed_forward_length", 32);
            g.AddUint32("dflash.attention.head_count", 8);
            g.AddUint32("dflash.attention.head_count_kv", 2);
            g.AddUint32("dflash.attention.key_length", 16);
            g.AddUint32("dflash.attention.value_length", 16);
            g.AddFloat32("dflash.attention.layer_norm_rms_epsilon", 1e-6f);
            g.AddFloat32("dflash.rope.freq_base", 10000f);
            g.AddUint32("dflash.block_size", 8);
            g.AddInt32Array("dflash.target_layers", new[] { 1, 2 });
            g.AddUint32("tokenizer.ggml.mask_token_id", 990);
            if (withSwa)
            {
                g.AddUint32("dflash.attention.sliding_window", 64);
                g.AddBoolArray("dflash.attention.sliding_window_pattern", new[] { true, true });
            }
            if (withSampleAnchor)
                g.AddString("dflash.sample_from_anchor", "false");
            if (withMarkov)
            {
                g.AddTensor("markov_w1.weight", 30, 8);    // 512 x 16 -> rank = Shape[0] = 8
                g.AddTensor("markov_w2.weight", 30, 8);
            }
            if (withSinks)
                g.AddTensor("blk.0.attn_sinks", 0, 8);
            return g;
        }

        [Fact]
        public void MarkovDrafter_DetectsRankSinksAndDefaults()
        {
            string path = BaseDraft(withMarkov: true, withSinks: true, withSwa: false, withSampleAnchor: false).Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.Equal(512, cfg.MarkovRank);
                Assert.True(cfg.HasAttentionSinks);
                Assert.True(cfg.SampleFromAnchor);
                Assert.Equal(1024, cfg.SlidingWindow);
                Assert.Equal(cfg.BlockSize, cfg.MaxDraftTokens);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void PlainDrafter_WithoutMarkov_DraftsBlockSizeMinusOne()
        {
            string path = BaseDraft(withMarkov: false, withSinks: false, withSwa: true, withSampleAnchor: false).Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.Equal(0, cfg.MarkovRank);
                Assert.False(cfg.HasAttentionSinks);
                Assert.Equal(64, cfg.SlidingWindow);
                Assert.Equal(cfg.BlockSize - 1, cfg.MaxDraftTokens);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SampleFromAnchorKey_IsHonored()
        {
            string path = BaseDraft(withMarkov: true, withSinks: false, withSwa: false, withSampleAnchor: true).Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.False(cfg.SampleFromAnchor);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void SwaPattern_DefaultsToAllSliding_ForDSparkFiles()
        {
            string path = BaseDraft(withMarkov: true, withSinks: false, withSwa: false, withSampleAnchor: false).Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.Equal(2, cfg.SwaPattern.Length);
                Assert.True(cfg.SwaPattern[0] && cfg.SwaPattern[1]);
            }
            finally { File.Delete(path); }
        }
    }
}