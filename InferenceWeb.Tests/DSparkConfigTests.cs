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
    /// sliding_window = 1024 for DSpark files, all-sliding pattern), the
    /// draft-count consequence (a Markov drafter that samples from the anchor
    /// keeps the whole block width; a bonus anchor's does not), and the
    /// Nemotron-3.5 exports' keys (a bool bonus anchor, sinks named .weight, a
    /// causal block, no window, YaRN).
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

        /// <summary>The Nemotron-3.5 DSpark export's keys as llama.cpp writes them: sample_from_anchor as a BOOL (a bonus
        /// anchor: B - 1 drafts a block), the sinks as blk.N.attn_sinks.weight, and a block-causal mask with no DSpine keys.</summary>
        [Fact]
        public void NemotronDSparkExport_BoolBonusAnchorSinksWeightAndCausalBlock()
        {
            var g = BaseDraft(withMarkov: true, withSinks: false, withSwa: true, withSampleAnchor: false);
            g.AddBool("dflash.sample_from_anchor", false);
            g.AddBool("dflash.attention.causal", true);
            g.AddTensor("blk.0.attn_sinks.weight", 0, 8);
            string path = g.Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.False(cfg.SampleFromAnchor);
                Assert.False(cfg.DraftsFromAnchorRow);
                Assert.Equal(cfg.BlockSize - 1, cfg.MaxDraftTokens);
                Assert.True(cfg.HasAttentionSinks);
                Assert.True(cfg.CausalBlock);
                Assert.False(cfg.IsDSpine);
                Assert.Equal(64, cfg.SlidingWindow);
            }
            finally { File.Delete(path); }
        }

        /// <summary>A block-causal mask on a DFlash2 drafter (trained non-causal) is still refused.</summary>
        [Fact]
        public void CausalBlockOnADFlash2Drafter_IsRefused()
        {
            var g = BaseDraft(withMarkov: false, withSinks: false, withSwa: true, withSampleAnchor: false);
            g.AddBool("dflash.attention.causal", true);
            g.AddUint32("dflash.selector_rank", 8);
            g.AddUint32("dflash.selector_top_k", 4);
            string path = g.Write();
            try
            {
                using var gguf = new GgufFile(path);
                var ex = Assert.Throws<NotSupportedException>(() => DFlashConfig.FromGguf(gguf));
                Assert.Contains("causal", ex.Message);
            }
            finally { File.Delete(path); }
        }

        /// <summary>The Nemotron-3.5 DFlash export: no sliding window (every layer attends the whole history, served through
        /// the ring with the trunk's context, at most FullAttentionWindowCap positions) and a YaRN rotary.</summary>
        [Fact]
        public void FullAttentionYarnDrafter_IsServedWithTheTrunksContext()
        {
            var g = BaseDraft(withMarkov: false, withSinks: false, withSwa: false, withSampleAnchor: false);
            g.AddString("dflash.rope.scaling.type", "yarn");
            g.AddFloat32("dflash.rope.scaling.factor", 128f);
            g.AddUint32("dflash.rope.scaling.original_context_length", 8192);
            string path = g.Write();
            try
            {
                using var gguf = new GgufFile(path);
                var cfg = DFlashConfig.FromGguf(gguf);
                Assert.True(cfg.FullAttention);
                Assert.Equal(DFlashConfig.FullAttentionWindowCap, cfg.SlidingWindow);
                Assert.True(cfg.HasYarn);
                Assert.Equal(1f / 128f, cfg.RopeFreqScale);
                Assert.Equal(8192, cfg.RopeOriginalContext);
                Assert.Equal(1f, cfg.YarnExtFactor);
                Assert.Equal(32f, cfg.YarnBetaFast);
                Assert.Equal(1f, cfg.YarnBetaSlow);
                cfg.ServeFullAttention(4096);
                Assert.Equal(4096, cfg.SlidingWindow);
                Assert.Equal(4096 + 32, cfg.RingRows);
                cfg.ServeFullAttention(1 << 20);
                Assert.Equal(DFlashConfig.FullAttentionWindowCap, cfg.SlidingWindow);
            }
            finally { File.Delete(path); }
        }

        [Fact]
        public void UnknownRopeScaling_IsRefused()
        {
            var g = BaseDraft(withMarkov: false, withSinks: false, withSwa: true, withSampleAnchor: false);
            g.AddString("dflash.rope.scaling.type", "longrope");
            g.AddFloat32("dflash.rope.scaling.factor", 4f);
            string path = g.Write();
            try
            {
                using var gguf = new GgufFile(path);
                Assert.Throws<NotSupportedException>(() => DFlashConfig.FromGguf(gguf));
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