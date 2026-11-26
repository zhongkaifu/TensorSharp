// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;
using System.IO;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>A small gemma4-assistant (MTP draft head) GGUF for <see cref="Gemma4DenseSyntheticModelBuilder"/>'s target: two
/// blocks of width 128 (a sliding-window one and a global one, the target's head widths) whose four query heads group onto
/// the donors' two KV heads, its own Q8_0 token table as the head, and the pre/post projections to the target's width. The
/// newer converter's spelling (<c>mtp.</c> prefix, projections without <c>nextn.</c>) when <see cref="MtpPrefix"/>.</summary>
internal sealed class Gemma4AssistantSyntheticModelBuilder
{
    public const int Width = 128;
    public const int Heads = 4;
    public const int Ff = 256;

    public static readonly bool[] Pattern = { true, false };

    public bool MtpPrefix { get; init; }

    public string Write(string path)
    {
        const string a = "gemma4-assistant";
        int target = Gemma4DenseSyntheticModelBuilder.Hidden, vocab = Gemma4DenseSyntheticModelBuilder.Vocab;
        int local = Gemma4DenseSyntheticModelBuilder.LocalHeadDim, global = Gemma4DenseSyntheticModelBuilder.GlobalHeadDim;
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Pattern.Length },
            new U32 { Key = $"{a}.embedding_length", V = Width },
            new U32 { Key = $"{a}.embedding_length_out", V = (uint)target },
            new U32 { Key = $"{a}.feed_forward_length", V = Ff },
            new U32 { Key = $"{a}.attention.head_count", V = Heads },
            new U32 { Key = $"{a}.attention.key_length", V = (uint)global },
            new U32 { Key = $"{a}.attention.key_length_swa", V = (uint)local },
            new BoolArr { Key = $"{a}.attention.sliding_window_pattern", V = Pattern },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
        };
        AddByteTokenizer(kv, vocab, bos: 1, eos: 2);
        string prefix = MtpPrefix ? "mtp." : "";
        string pre = MtpPrefix ? "mtp.pre_projection.weight" : "nextn.pre_projection.weight";
        string post = MtpPrefix ? "mtp.post_projection.weight" : "nextn.post_projection.weight";
        var t = new List<Tensor>
        {
            Q8(Gen(prefix + "token_embd.weight", 0.08f, Width, vocab)),
            GenAround(prefix + "output_norm.weight", 1f, 0.1f, Width),
            Q8(Gen(pre, 0.05f, 2 * target, Width)),
            Q8(Gen(post, 0.06f, Width, target)),
        };
        for (int l = 0; l < Pattern.Length; l++)
        {
            string p = $"{prefix}blk.{l}.";
            int hd = Pattern[l] ? local : global;
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Width));
            t.Add(Q8(Gen(p + "attn_q.weight", 0.06f, Width, Heads * hd)));
            t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.05f, hd));
            t.Add(Q8(Gen(p + "attn_output.weight", 0.04f, Heads * hd, Width)));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Width));
            t.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Width));
            t.Add(Q8(Gen(p + "ffn_gate.weight", 0.06f, Width, Ff)));
            t.Add(Q8(Gen(p + "ffn_up.weight", 0.06f, Width, Ff)));
            t.Add(Q8(Gen(p + "ffn_down.weight", 0.06f, Ff, Width)));
            t.Add(GenAround(p + "post_ffw_norm.weight", 0.5f, 0.05f, Width));
            t.Add(new Tensor { Name = p + "layer_output_scale.weight", Dims = new[] { 1UL }, Data = new[] { 0.9f } });
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private static Tensor Q8(Tensor x)
    {
        x.Type = GgmlType.Q8_0;
        return x;
    }
}
