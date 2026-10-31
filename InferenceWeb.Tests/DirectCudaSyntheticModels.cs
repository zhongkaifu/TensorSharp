// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tiny, deterministic GGUFs of the three per-op families the direct-CUDA prefill plan targets
// besides Nemotron (which has NemotronSyntheticModelBuilder): gpt-oss, Qwen3.5 MoE (qwen35moe)
// and Gemma-4 MoE. Each mirrors the tensor list, tensor types and metadata keys of the real
// checkpoint it stands for (gpt-oss-20b MXFP4, Qwen3.5-35B-A3B UD-IQ4_XS, gemma-4-26B-A4B Q4_0;
// the headers were read from the A40 VMs' files) at small widths, keeping the shapes that pick
// kernel routes: head dims, GQA group sizes, sliding windows, sinks, K-tails that are not
// multiples of 128, the expert quant types and top-k. Weights are name-seeded, so two files
// written with the same options are bit-identical.
using System;
using System.Collections.Generic;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>gpt-oss: alternating sliding-window (even layers) and full attention with sinks and
/// biases, head dim 64 at GQA group 8, yarn RoPE, MXFP4 routed experts with biases, top-4.</summary>
internal sealed class GptOssSyntheticModelBuilder
{
    public const int Hidden = 288;          // 9 Q8_0 / MXFP4 blocks: K % 128 != 0, like 2880
    public const int NumHeads = 16;
    public const int NumKvHeads = 2;        // group 8, as gpt-oss-20b (64 / 8)
    public const int HeadDim = 64;
    public const int Vocab = 256;

    public int Layers { get; init; } = 4;
    public int Experts { get; init; } = 8;
    public int TopK { get; init; } = 4;
    public int ExpertFf { get; init; } = 288;
    public int SlidingWindow { get; init; } = 32;
    public GgmlType ExpertType { get; init; } = GgmlType.MXFP4;
    /// <summary>The per-head attention sinks are drawn around this value with <see cref="SinkSpread"/>.
    /// The defaults reproduce the original file; a sink well above the typical q.k score takes a
    /// real share of the softmax, so a path that drops or misapplies the sinks moves the logits
    /// far more than the backends' rounding does (the tests use <see cref="SensitiveSinks"/>).</summary>
    public float SinkCenter { get; init; }
    public float SinkSpread { get; init; } = 0.5f;

    /// <summary>The sink setting the direct-CUDA model tests use (see DirectCudaPrefillModelTests).</summary>
    public static GptOssSyntheticModelBuilder SensitiveSinks() => new() { SinkCenter = 2f, SinkSpread = 1f };

    public string Write(string path)
    {
        const string a = "gpt-oss";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = (uint)ExpertFf },
            new U32 { Key = $"{a}.attention.head_count", V = NumHeads },
            new U32 { Key = $"{a}.attention.head_count_kv", V = NumKvHeads },
            new Str { Key = $"{a}.rope.scaling.type", V = "yarn" },
            new F32 { Key = $"{a}.rope.scaling.factor", V = 32f },
            new U32 { Key = $"{a}.rope.scaling.original_context_length", V = 4096 },
            new F32 { Key = $"{a}.rope.scaling.yarn_beta_fast", V = 32f },
            new F32 { Key = $"{a}.rope.scaling.yarn_beta_slow", V = 1f },
            new F32 { Key = $"{a}.rope.freq_base", V = 150000f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-5f },
            new U32 { Key = $"{a}.expert_count", V = (uint)Experts },
            new U32 { Key = $"{a}.expert_used_count", V = (uint)TopK },
            new U32 { Key = $"{a}.attention.key_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.sliding_window", V = (uint)SlidingWindow },
            new U32 { Key = $"{a}.expert_feed_forward_length", V = (uint)ExpertFf },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        int qDim = NumHeads * HeadDim, kvDim = NumKvHeads * HeadDim;
        var t = new List<Tensor>
        {
            Q8(Gen("token_embd.weight", 0.5f, Hidden, Vocab)),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            Q8(Gen("output.weight", 0.08f, Hidden, Vocab)),
        };
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(Q8(Gen(p + "attn_q.weight", 0.06f, Hidden, qDim)));
            t.Add(Gen(p + "attn_q.bias", 0.05f, qDim));
            t.Add(Q8(Gen(p + "attn_k.weight", 0.06f, Hidden, kvDim)));
            t.Add(Gen(p + "attn_k.bias", 0.05f, kvDim));
            t.Add(Q8(Gen(p + "attn_v.weight", 0.06f, Hidden, kvDim)));
            t.Add(Gen(p + "attn_v.bias", 0.05f, kvDim));
            t.Add(Q8(Gen(p + "attn_output.weight", 0.04f, qDim, Hidden)));
            t.Add(Gen(p + "attn_output.bias", 0.05f, Hidden));
            t.Add(GenAround(p + "attn_sinks.weight", SinkCenter, SinkSpread, NumHeads));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
            t.Add(Gen(p + "ffn_gate_inp.weight", 0.3f, Hidden, Experts));
            t.Add(Gen(p + "ffn_gate_inp.bias", 0.1f, Experts));
            t.Add(Expert(p + "ffn_gate_exps.weight", 0.06f, Hidden, ExpertFf));
            t.Add(Gen(p + "ffn_gate_exps.bias", 0.05f, ExpertFf, Experts));
            t.Add(Expert(p + "ffn_up_exps.weight", 0.06f, Hidden, ExpertFf));
            t.Add(Gen(p + "ffn_up_exps.bias", 0.05f, ExpertFf, Experts));
            t.Add(Expert(p + "ffn_down_exps.weight", 0.06f, ExpertFf, Hidden));
            t.Add(Gen(p + "ffn_down_exps.bias", 0.05f, Hidden, Experts));
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private Tensor Expert(string name, float scale, int inDim, int outDim) => ExpertType == GgmlType.Q8_0
        ? Q8(Gen(name, scale, inDim, outDim, Experts))
        : Blocks(name, ExpertType, scale, inDim, outDim, Experts);

    internal static Tensor Q8(Tensor x)
    {
        x.Type = GgmlType.Q8_0;
        return x;
    }
}

/// <summary>Qwen3.5 MoE (qwen35moe): three Gated DeltaNet layers then one gated full-attention
/// layer (full_attention_interval 4), head dim 256 at GQA group 8 with q/k norms and M-RoPE
/// sections, GDN head dims 128, IQ3_S gate/up and IQ4_XS down experts (the UD-IQ4_XS mix) with a
/// sigmoid-gated shared expert, top-4 of 16.</summary>
internal sealed class Qwen35MoeSyntheticModelBuilder
{
    public const int Hidden = 256;           // IQ3_S / IQ4_XS rows need multiples of 256
    public const int NumHeads = 8;
    public const int NumKvHeads = 1;         // group 8, as the 35B-A3B (16 / 2)
    public const int HeadDim = 256;
    public const int Vocab = 256;
    public const int SsmKHeads = 2;          // ssm.group_count
    public const int SsmVHeads = 4;          // ssm.time_step_rank
    public const int SsmHeadDim = 128;       // ssm.state_size (k) and inner / v heads (v)
    public const int SsmInner = SsmVHeads * SsmHeadDim;
    public const int SsmConv = 4;

    public int Layers { get; init; } = 4;
    public int Experts { get; init; } = 16;
    public int TopK { get; init; } = 4;
    public int ExpertFf { get; init; } = 256;
    public int SharedFf { get; init; } = 256;

    public string Write(string path)
    {
        const string a = "qwen35moe";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.attention.head_count", V = NumHeads },
            new U32 { Key = $"{a}.attention.head_count_kv", V = NumKvHeads },
            new I32Arr { Key = $"{a}.rope.dimension_sections", V = new[] { 11, 11, 10, 0 } },
            new F32 { Key = $"{a}.rope.freq_base", V = 10000000f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = $"{a}.expert_count", V = (uint)Experts },
            new U32 { Key = $"{a}.expert_used_count", V = (uint)TopK },
            new U32 { Key = $"{a}.attention.key_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = HeadDim },
            new U32 { Key = $"{a}.expert_feed_forward_length", V = (uint)ExpertFf },
            new U32 { Key = $"{a}.expert_shared_feed_forward_length", V = (uint)SharedFf },
            new U32 { Key = $"{a}.ssm.conv_kernel", V = SsmConv },
            new U32 { Key = $"{a}.ssm.state_size", V = SsmHeadDim },
            new U32 { Key = $"{a}.ssm.group_count", V = SsmKHeads },
            new U32 { Key = $"{a}.ssm.time_step_rank", V = SsmVHeads },
            new U32 { Key = $"{a}.ssm.inner_size", V = SsmInner },
            new U32 { Key = $"{a}.full_attention_interval", V = 4 },
            new U32 { Key = $"{a}.rope.dimension_count", V = 64 },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        var t = new List<Tensor>
        {
            GptOssSyntheticModelBuilder.Q8(Gen("token_embd.weight", 0.5f, Hidden, Vocab)),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            Blocks("output.weight", GgmlType.Q6_K, 0.08f, Hidden, Vocab),
        };
        int qkvDim = 2 * SsmKHeads * SsmHeadDim + SsmInner;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
            if ((l + 1) % 4 != 0)
            {
                t.Add(Q8(p + "attn_qkv.weight", 0.06f, Hidden, qkvDim));
                t.Add(Q8(p + "attn_gate.weight", 0.06f, Hidden, SsmInner));
                t.Add(Q8(p + "ssm_alpha.weight", 0.06f, Hidden, SsmVHeads));
                t.Add(Q8(p + "ssm_beta.weight", 0.06f, Hidden, SsmVHeads));
                t.Add(GenAround(p + "ssm_a", -1f, 0.25f, SsmVHeads));
                t.Add(Gen(p + "ssm_dt.bias", 0.5f, SsmVHeads));
                t.Add(Gen(p + "ssm_conv1d.weight", 0.3f, SsmConv, qkvDim));
                t.Add(GenAround(p + "ssm_norm.weight", 1f, 0.1f, SsmHeadDim));
                t.Add(Q8(p + "ssm_out.weight", 0.04f, SsmInner, Hidden));
            }
            else
            {
                t.Add(Q8(p + "attn_q.weight", 0.06f, Hidden, 2 * NumHeads * HeadDim));   // Q and its gate
                t.Add(Q8(p + "attn_k.weight", 0.06f, Hidden, NumKvHeads * HeadDim));
                t.Add(Q8(p + "attn_v.weight", 0.06f, Hidden, NumKvHeads * HeadDim));
                t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.1f, HeadDim));
                t.Add(GenAround(p + "attn_k_norm.weight", 1f, 0.1f, HeadDim));
                t.Add(Q8(p + "attn_output.weight", 0.04f, NumHeads * HeadDim, Hidden));
            }
            t.Add(Gen(p + "ffn_gate_inp.weight", 0.3f, Hidden, Experts));
            t.Add(Blocks(p + "ffn_gate_exps.weight", GgmlType.IQ3_S, 0.06f, Hidden, ExpertFf, Experts));
            t.Add(Blocks(p + "ffn_up_exps.weight", GgmlType.IQ3_S, 0.06f, Hidden, ExpertFf, Experts));
            t.Add(Blocks(p + "ffn_down_exps.weight", GgmlType.IQ4_XS, 0.06f, ExpertFf, Hidden, Experts));
            t.Add(Gen(p + "ffn_gate_inp_shexp.weight", 0.3f, Hidden));
            t.Add(Q8(p + "ffn_gate_shexp.weight", 0.06f, Hidden, SharedFf));
            t.Add(Q8(p + "ffn_up_shexp.weight", 0.06f, Hidden, SharedFf));
            t.Add(Q8(p + "ffn_down_shexp.weight", 0.06f, SharedFf, Hidden));
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private static Tensor Q8(string name, float scale, params int[] dims) => GptOssSyntheticModelBuilder.Q8(Gen(name, scale, dims));
}

/// <summary>Dense Qwen3.5 (qwen35, the 0.8B layout that Stage 0 calls qwen08): three Gated DeltaNet
/// layers then one gated full-attention layer, head dim 256 at GQA group 4 (8 / 2), GDN with as
/// many value heads as key heads (16 / 16 in the 0.8B), a dense SwiGLU FFN and a tied output head.
/// Every projection is BF16 (the BF16 checkpoint takes its own direct-CUDA matmul branch, ahead of
/// every quantized route), norms and the conv kernel F32.</summary>
internal sealed class Qwen35DenseSyntheticModelBuilder
{
    public const int Hidden = 256;
    public const int NumHeads = 8;
    public const int NumKvHeads = 2;         // group 4, as the 0.8B (8 / 2)
    public const int HeadDim = 256;
    public const int Vocab = 256;
    public const int SsmKHeads = 2;          // ssm.group_count
    public const int SsmVHeads = 2;          // ssm.time_step_rank (== group_count in the 0.8B)
    public const int SsmHeadDim = 128;
    public const int SsmInner = SsmVHeads * SsmHeadDim;
    public const int SsmConv = 4;

    public int Layers { get; init; } = 4;
    public int Ff { get; init; } = 448;      // 3584 / 8: not a multiple of 128 or 256
    public GgmlType WeightType { get; init; } = GgmlType.BF16;
    /// <summary>Scale of the token embedding, which is also the output head (tied).</summary>
    public float EmbeddingScale { get; init; } = 0.5f;
    /// <summary>Write an untied output.weight (the 0.8B ties the head to the embedding).</summary>
    public bool UntiedOutput { get; init; }

    /// <summary>A file whose greedy continuation depends on the history: with the tied head these
    /// small files repeat their last token (0/12 argmax changes when only the history changes);
    /// untied, 7/12 and varied streams.</summary>
    public static Qwen35DenseSyntheticModelBuilder ContextSensitive() => new() { UntiedOutput = true };

    public string Write(string path)
    {
        const string a = "qwen35";
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = (uint)Ff },
            new U32 { Key = $"{a}.attention.head_count", V = NumHeads },
            new U32 { Key = $"{a}.attention.head_count_kv", V = NumKvHeads },
            new I32Arr { Key = $"{a}.rope.dimension_sections", V = new[] { 11, 11, 10, 0 } },
            new F32 { Key = $"{a}.rope.freq_base", V = 10000000f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = $"{a}.attention.key_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = HeadDim },
            new U32 { Key = $"{a}.ssm.conv_kernel", V = SsmConv },
            new U32 { Key = $"{a}.ssm.state_size", V = SsmHeadDim },
            new U32 { Key = $"{a}.ssm.group_count", V = SsmKHeads },
            new U32 { Key = $"{a}.ssm.time_step_rank", V = SsmVHeads },
            new U32 { Key = $"{a}.ssm.inner_size", V = SsmInner },
            new U32 { Key = $"{a}.full_attention_interval", V = 4 },
            new U32 { Key = $"{a}.rope.dimension_count", V = 64 },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        var t = new List<Tensor>
        {
            W(Gen("token_embd.weight", EmbeddingScale, Hidden, Vocab)),   // tied output head, as the 0.8B
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
        };
        if (UntiedOutput)
            t.Add(W(Gen("output.weight", 0.08f, Hidden, Vocab)));
        int qkvDim = 2 * SsmKHeads * SsmHeadDim + SsmInner;
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
            if ((l + 1) % 4 != 0)
            {
                t.Add(W(Gen(p + "attn_qkv.weight", 0.06f, Hidden, qkvDim)));
                t.Add(W(Gen(p + "attn_gate.weight", 0.06f, Hidden, SsmInner)));
                t.Add(W(Gen(p + "ssm_alpha.weight", 0.06f, Hidden, SsmVHeads)));
                t.Add(W(Gen(p + "ssm_beta.weight", 0.06f, Hidden, SsmVHeads)));
                t.Add(GenAround(p + "ssm_a", -1f, 0.25f, SsmVHeads));
                t.Add(Gen(p + "ssm_dt.bias", 0.5f, SsmVHeads));
                t.Add(Gen(p + "ssm_conv1d.weight", 0.3f, SsmConv, qkvDim));
                t.Add(GenAround(p + "ssm_norm.weight", 1f, 0.1f, SsmHeadDim));
                t.Add(W(Gen(p + "ssm_out.weight", 0.04f, SsmInner, Hidden)));
            }
            else
            {
                t.Add(W(Gen(p + "attn_q.weight", 0.06f, Hidden, 2 * NumHeads * HeadDim)));   // Q and its gate
                t.Add(W(Gen(p + "attn_k.weight", 0.06f, Hidden, NumKvHeads * HeadDim)));
                t.Add(W(Gen(p + "attn_v.weight", 0.06f, Hidden, NumKvHeads * HeadDim)));
                t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.1f, HeadDim));
                t.Add(GenAround(p + "attn_k_norm.weight", 1f, 0.1f, HeadDim));
                t.Add(W(Gen(p + "attn_output.weight", 0.04f, NumHeads * HeadDim, Hidden)));
            }
            t.Add(W(Gen(p + "ffn_gate.weight", 0.06f, Hidden, Ff)));
            t.Add(W(Gen(p + "ffn_up.weight", 0.06f, Hidden, Ff)));
            t.Add(W(Gen(p + "ffn_down.weight", 0.05f, Ff, Hidden)));
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private Tensor W(Tensor x)
    {
        x.Type = WeightType;
        return x;
    }
}

/// <summary>Gemma-4 MoE (the 26B-A4B layout): sliding-window layers (head dim 256, GQA group 2)
/// and global layers (head dim 512, GQA group 8, K = V, proportional RoPE over a quarter of the
/// head), q/k norms, a dense GELU FFN beside a scaled top-k MoE with fused gate_up Q4_0 experts
/// whose down K (96) is not a multiple of 128, per-layer output scales and a softcapped tied head.</summary>
internal sealed class Gemma4MoeSyntheticModelBuilder
{
    public const int Hidden = 256;
    public const int NumHeads = 8;
    public const int LocalKvHeads = 4;       // group 2, as the 26B's 16 / 8
    public const int GlobalKvHeads = 1;      // group 8, as the 26B's 16 / 2
    public const int LocalHeadDim = 256;
    public const int GlobalHeadDim = 512;
    public const int Vocab = 256;
    public const int DenseFf = 128;

    /// <summary>Sliding (true) or global (false) per layer; the 26B repeats five local, one global.</summary>
    public bool[] Pattern { get; init; } = { true, true, false, true, true, false };
    public int Experts { get; init; } = 16;
    public int TopK { get; init; } = 4;
    public int ExpertFf { get; init; } = 96;
    public int SlidingWindow { get; init; } = 32;
    /// <summary>Scale of the (tied) token embedding. Gemma multiplies the embedding by sqrt(hidden)
    /// before the first layer, so at the default 0.5 the embedding row dominates the residual
    /// stream through all six layers and the next-token argmax depends almost only on the last
    /// token (the P0 review measured 0/6 argmax changes when only the history changed). A smaller
    /// embedding lets attention and the MoE move the argmax with the context, which a test that
    /// compares greedy streams needs (<see cref="ContextSensitive"/>).</summary>
    public float EmbeddingScale { get; init; } = 0.5f;
    /// <summary>Centre of post_attention_norm (the attention block's output weight; default 0.3).
    /// Attention is the only path by which earlier tokens reach the next-token logits.</summary>
    public float AttentionPostNorm { get; init; } = 0.3f;
    /// <summary>Centre of attn_k_norm (default 0.12, the 26B's): larger sharpens the softmax.</summary>
    public float KNorm { get; init; } = 0.12f;
    /// <summary>Write an untied Q8_0 output.weight (the real Gemma-4 files tie the head to the
    /// embedding; with the tie, the greedy continuation of these small files settles on repeating
    /// one token whatever the context).</summary>
    public bool UntiedOutput { get; init; }

    /// <summary>A file whose greedy continuation depends on the history, not only on the last
    /// token: global layers only (with the sliding window the default decode is D3 and repeats
    /// nothing), an untied head, a small embedding and a stronger attention output. Measured: the
    /// argmax moves with the history in 8/12 probes and the greedy streams vary (the default file:
    /// 0/12, every stream one repeated token).</summary>
    public static Gemma4MoeSyntheticModelBuilder ContextSensitive() => new()
    {
        Pattern = new[] { false, false, false, false, false, false },
        UntiedOutput = true,
        EmbeddingScale = 0.05f,
        AttentionPostNorm = 1f,
    };

    public string Write(string path)
    {
        const string a = "gemma4";
        int layers = Pattern.Length;
        var kvHeads = new int[layers];
        for (int l = 0; l < layers; l++)
            kvHeads[l] = Pattern[l] ? LocalKvHeads : GlobalKvHeads;
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = DenseFf },
            new U32 { Key = $"{a}.attention.head_count", V = NumHeads },
            new I32Arr { Key = $"{a}.attention.head_count_kv", V = kvHeads },
            new F32 { Key = $"{a}.rope.freq_base", V = 1000000f },
            new F32 { Key = $"{a}.rope.freq_base_swa", V = 10000f },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = $"{a}.expert_count", V = (uint)Experts },
            new U32 { Key = $"{a}.expert_used_count", V = (uint)TopK },
            new U32 { Key = $"{a}.attention.key_length", V = GlobalHeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = GlobalHeadDim },
            new F32 { Key = $"{a}.final_logit_softcapping", V = 30f },
            new U32 { Key = $"{a}.attention.sliding_window", V = (uint)SlidingWindow },
            new U32 { Key = $"{a}.attention.shared_kv_layers", V = 0 },
            new U32 { Key = $"{a}.embedding_length_per_layer_input", V = 0 },
            new BoolArr { Key = $"{a}.attention.sliding_window_pattern", V = Pattern },
            new U32 { Key = $"{a}.attention.key_length_swa", V = LocalHeadDim },
            new U32 { Key = $"{a}.attention.value_length_swa", V = LocalHeadDim },
            new U32 { Key = $"{a}.expert_feed_forward_length", V = (uint)ExpertFf },
            new U32 { Key = $"{a}.rope.dimension_count", V = GlobalHeadDim },
            new U32 { Key = $"{a}.rope.dimension_count_swa", V = LocalHeadDim },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        // Proportional RoPE: the first quarter of the global head's frequency pairs rotate,
        // the rest are pinned by a 1e30 divisor (the 26B's rope_freqs.weight).
        var ropeFreqs = new float[GlobalHeadDim / 2];
        for (int i = 0; i < ropeFreqs.Length; i++)
            ropeFreqs[i] = i < ropeFreqs.Length / 4 ? 1f : 1e30f;
        var t = new List<Tensor>
        {
            GptOssSyntheticModelBuilder.Q8(Gen("token_embd.weight", EmbeddingScale, Hidden, Vocab)),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            new Tensor { Name = "rope_freqs.weight", Dims = new[] { (ulong)ropeFreqs.Length }, Data = ropeFreqs },
        };
        if (UntiedOutput)
            t.Add(GptOssSyntheticModelBuilder.Q8(Gen("output.weight", 0.08f, Hidden, Vocab)));
        for (int l = 0; l < layers; l++)
        {
            string p = $"blk.{l}.";
            bool local = Pattern[l];
            int hd = local ? LocalHeadDim : GlobalHeadDim;
            int kvh = local ? LocalKvHeads : GlobalKvHeads;
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(Blocks(p + "attn_q.weight", GgmlType.Q4_0, 0.06f, Hidden, NumHeads * hd));
            t.Add(Blocks(p + "attn_k.weight", GgmlType.Q4_0, 0.06f, Hidden, kvh * hd));
            if (local)
                t.Add(Blocks(p + "attn_v.weight", GgmlType.Q4_0, 0.06f, Hidden, kvh * hd));   // global layers: V = K
            // Gemma 4 attends with scale 1 after q/k norms; the 26B's k_norm is 0.122 (q_norm
            // 1.02), which keeps q.k at a few units. With both at 1 the scores of a 256-wide head
            // spread to about 16, the softmax is one-hot and a rounding difference picks another key.
            t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.05f, hd));
            t.Add(GenAround(p + "attn_k_norm.weight", KNorm, KNorm / 12f, hd));
            t.Add(Blocks(p + "attn_output.weight", GgmlType.Q4_0, 0.04f, NumHeads * hd, Hidden));
            // Post-norm weights well below 1 and layer scales near 1 keep each block's
            // contribution small against the embedding-led residual stream, so a router
            // near-tie that one backend resolves differently moves the logits by a few
            // percent instead of flipping them (with the 26B's 0.1-0.66 layer scales the
            // residual decays and the last MoE choice decides everything).
            t.Add(GenAround(p + "post_attention_norm.weight", AttentionPostNorm, AttentionPostNorm / 10f, Hidden));
            t.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(Blocks(p + "ffn_gate.weight", GgmlType.Q4_0, 0.06f, Hidden, DenseFf));
            t.Add(Blocks(p + "ffn_up.weight", GgmlType.Q4_0, 0.06f, Hidden, DenseFf));
            t.Add(Blocks(p + "ffn_down.weight", GgmlType.Q4_0, 0.06f, DenseFf, Hidden));
            t.Add(GenAround(p + "post_ffw_norm.weight", 0.3f, 0.03f, Hidden));
            t.Add(GenAround(p + "post_ffw_norm_1.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "post_ffw_norm_2.weight", 0.6f, 0.06f, Hidden));
            t.Add(GenAround(p + "pre_ffw_norm_2.weight", 1f, 0.1f, Hidden));
            t.Add(Gen(p + "ffn_gate_inp.weight", 0.3f, Hidden, Experts));
            t.Add(GenAround(p + "ffn_gate_inp.scale", 1f, 0.1f, Hidden));
            t.Add(Blocks(p + "ffn_gate_up_exps.weight", GgmlType.Q4_0, 0.06f, Hidden, 2 * ExpertFf, Experts));
            t.Add(Blocks(p + "ffn_down_exps.weight", GgmlType.Q4_0, 0.06f, ExpertFf, Hidden, Experts));
            t.Add(GenAround(p + "ffn_down_exps.scale", 1f, 0.02f, Experts));
            t.Add(new Tensor { Name = p + "layer_output_scale.weight", Dims = new[] { 1UL }, Data = new[] { local ? 0.95f : 0.9f } });
        }
        SyntheticGguf.Write(path, kv, t);
        return path;
    }
}
