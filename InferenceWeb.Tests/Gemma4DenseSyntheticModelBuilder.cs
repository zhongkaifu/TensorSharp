// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A small dense Gemma-4 GGUF with the E2B/E4B layout (see the summary below); used by
// Gemma4DenseBatchedForwardParityTests. Weights are name-seeded, so two files are bit-identical.
using System.Collections.Generic;
using System.IO;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>A small dense Gemma-4 with the E2B/E4B layout: sliding-window layers (head dim 256,
/// window 208) and global layers (head dim 512, proportional RoPE over the first quarter of its
/// pairs via rope_freqs.weight: E4B's 64 ones and 192 x 1e30), two KV heads everywhere, q/k
/// norms, an unweighted V norm, a GELU FFN, per-layer embeddings (PLE), shared-KV layers (the
/// last two take the K/V of the last earlier layer of their kind) and per-layer output scales;
/// Q4_0/Q8_0 weights (a global layer's Q/K/V F32, see below), a softcapped tied head. Three of the eight
/// layers are global (E4B: one in six) so the global layers carry weight in the logits.</summary>
internal sealed class Gemma4DenseSyntheticModelBuilder
{
    public const int Hidden = 384;          // not a head dim, as E4B's 2560
    public const int NumHeads = 8;
    public const int KvHeads = 2;
    public const int LocalHeadDim = 256;
    public const int GlobalHeadDim = 512;
    public const int Vocab = 256;
    public const int Ff = 256;
    public const int PleDim = 32;
    public const int SharedKvLayers = 2;
    public const int SlidingWindow = 208;

    /// <summary>Sliding (true) or global (false) per layer; layers 6 (local) and 7 (global) share
    /// the K/V of layers 4 and 5.</summary>
    public static readonly bool[] Pattern = { true, true, false, true, true, false, true, false };

    public string Write(string path)
    {
        const string a = "gemma4";
        int layers = Pattern.Length;
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = Ff },
            new U32 { Key = $"{a}.attention.head_count", V = NumHeads },
            new U32 { Key = $"{a}.attention.head_count_kv", V = KvHeads },
            new F32 { Key = $"{a}.rope.freq_base", V = 1000000f },
            new F32 { Key = $"{a}.rope.freq_base_swa", V = 10000f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = $"{a}.attention.key_length", V = GlobalHeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = GlobalHeadDim },
            new F32 { Key = $"{a}.final_logit_softcapping", V = 30f },
            new U32 { Key = $"{a}.attention.sliding_window", V = SlidingWindow },
            new U32 { Key = $"{a}.attention.shared_kv_layers", V = SharedKvLayers },
            new U32 { Key = $"{a}.embedding_length_per_layer_input", V = PleDim },
            new BoolArray { Key = $"{a}.attention.sliding_window_pattern", V = Pattern },
            new U32 { Key = $"{a}.attention.key_length_swa", V = LocalHeadDim },
            new U32 { Key = $"{a}.attention.value_length_swa", V = LocalHeadDim },
            new U32 { Key = $"{a}.rope.dimension_count", V = GlobalHeadDim },
            new U32 { Key = $"{a}.rope.dimension_count_swa", V = LocalHeadDim },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        var ropeFreqs = new float[GlobalHeadDim / 2];
        for (int i = 0; i < ropeFreqs.Length; i++)
            ropeFreqs[i] = i < ropeFreqs.Length / 4 ? 1f : 1e30f;
        var t = new List<Tensor>
        {
            Q8(Gen("token_embd.weight", 0.05f, Hidden, Vocab)),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            new Tensor { Name = "rope_freqs.weight", Dims = new[] { (ulong)ropeFreqs.Length }, Data = ropeFreqs },
            Q8(Gen("per_layer_token_embd.weight", 0.05f, PleDim * layers, Vocab)),
            Q8(Gen("per_layer_model_proj.weight", 0.06f, Hidden, PleDim * layers)),
            GenAround("per_layer_proj_norm.weight", 1f, 0.1f, PleDim),
        };
        for (int l = 0; l < layers; l++)
        {
            string p = $"blk.{l}.";
            int hd = Pattern[l] ? LocalHeadDim : GlobalHeadDim;
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            // A global layer's Q/K/V are zero-mean F32. With Q4_0 blocks of random codes (mean
            // code - 8 = -0.5) every row shares one bias direction, the global scores barely
            // depend on the keys and no RoPE change moves the logits: the single-sequence logits
            // with and without the rope_freqs factors were 0-3e-3 apart at 41-300 tokens; with
            // this file they are 4.1e-2, 0.15 and 0.21 apart at 60, 120 and 205 tokens. (V is F32
            // too: the load-time QKV fusion takes F32 Q/K with a V of another storage as V = K.)
            if (!Pattern[l])
            {
                t.Add(Gen(p + "attn_q.weight", 0.06f, Hidden, NumHeads * hd));
                t.Add(Gen(p + "attn_k.weight", 0.06f, Hidden, KvHeads * hd));
                t.Add(Gen(p + "attn_v.weight", 0.06f, Hidden, KvHeads * hd));
            }
            else
            {
                t.Add(Blocks(p + "attn_q.weight", GgmlType.Q4_0, 0.06f, Hidden, NumHeads * hd));
                t.Add(Blocks(p + "attn_k.weight", GgmlType.Q4_0, 0.06f, Hidden, KvHeads * hd));
                t.Add(Blocks(p + "attn_v.weight", GgmlType.Q4_0, 0.06f, Hidden, KvHeads * hd));
            }
            // Gemma attends at scale 1 after the q/k norms; E4B-like small k_norm keeps the
            // scores at a few units, so attention mixes many positions instead of one.
            t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.05f, hd));
            t.Add(GenAround(p + "attn_k_norm.weight", 0.12f, 0.01f, hd));
            t.Add(Blocks(p + "attn_output.weight", GgmlType.Q4_0, 0.04f, NumHeads * hd, Hidden));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(Blocks(p + "ffn_gate.weight", GgmlType.Q4_0, 0.06f, Hidden, Ff));
            t.Add(Blocks(p + "ffn_up.weight", GgmlType.Q4_0, 0.06f, Hidden, Ff));
            t.Add(Blocks(p + "ffn_down.weight", GgmlType.Q4_0, 0.06f, Ff, Hidden));
            t.Add(GenAround(p + "post_ffw_norm.weight", 0.3f, 0.03f, Hidden));
            t.Add(Q8(Gen(p + "inp_gate.weight", 0.06f, Hidden, PleDim)));
            t.Add(Q8(Gen(p + "proj.weight", 0.06f, PleDim, Hidden)));
            t.Add(GenAround(p + "post_norm.weight", 0.3f, 0.03f, Hidden));
            t.Add(new Tensor { Name = p + "layer_output_scale.weight", Dims = new[] { 1UL }, Data = new[] { Pattern[l] ? 0.95f : 0.9f } });
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private static Tensor Q8(Tensor x)
    {
        x.Type = GgmlType.Q8_0;
        return x;
    }

    /// <summary>A GGUF array of bools (gemma4.attention.sliding_window_pattern).</summary>
    private sealed class BoolArray : Kv
    {
        public IReadOnlyList<bool> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(7u); w.Write((ulong)V.Count);
            foreach (bool v in V) w.Write(v);
        }
    }
}
