// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>
/// A small Thinking Machines Inkling 2 checkpoint (GGUF <c>inkling2_loop</c>, or <c>inkling2</c> without the loop) in F32,
/// with the real checkpoint's tensor names, shapes (ne0 first) and metadata keys (llama.cpp's src/models/inkling2.cpp and
/// inkling2-loop.cpp, convert_hf_to_gguf's Inkling2Model): no RoPE, sliding and global attention layers with the
/// relative-position bias (a zero column past the extent), per-query sinks, Q/K/R L2 clips, the global layers' tau and
/// dynamic conv taps, short convolutions on K, V, the attention output and the MoE output, block attention residuals,
/// sigmoid-routed experts with a selection bias and a fused gate_up, SITU, a padded vocabulary and the final softcap, and
/// the looped core with cross-pass messages and their controllers.
/// <para>
/// The default dims are chosen so a prompt of ~20 tokens reaches every mechanism: the window (6) slides, distances past
/// the global layers' extent (8) hit the zero column, tau departs from 1 at position 4 (floor 4), the clips cut some rows
/// and not others, the selection bias changes some routings, and the loop core [2, 6) runs three passes. Weights depend
/// only on their names and <see cref="Seed"/> (<see cref="SyntheticGguf.Gen"/>), so the loop and plain variants share every
/// tensor they both have.
/// </para>
/// </summary>
internal sealed class Inkling2SyntheticModelBuilder
{
    public int Hidden { get; init; } = 64;
    public int Heads { get; init; } = 4;
    public int KvHeads { get; init; } = 2;
    public int HeadDim { get; init; } = 16;
    /// <summary>Feed-forward width of each expert (gate and up are each this many rows of ffn_gate_up_exps).</summary>
    public int ExpertFf { get; init; } = 16;
    public int Experts { get; init; } = 8;
    public int ExpertsUsed { get; init; } = 3;
    /// <summary>The tokenizer's size, <c>vocab_size_unpadded</c>: the logits the model returns.</summary>
    public int Vocab { get; init; } = 61;
    /// <summary>Rows of token_embd and output, <c>vocab_size</c>.</summary>
    public int VocabPadded { get; init; } = 64;
    /// <summary>Layer kinds by physical index: 's' sliding window, 'g' global.</summary>
    public string Layout { get; init; } = "sgsgsgsg";
    public int Window { get; init; } = 6;
    public int BlockSize { get; init; } = 2;
    public int ConvTaps { get; init; } = 4;
    public int RelativeLength { get; init; } = 4;
    /// <summary>The global layers' bias extent; the sliding layers' is <see cref="Window"/>.</summary>
    public int RelativeExtent { get; init; } = 8;
    public float QkClip { get; init; } = 5.0f;
    public float RelativeClip { get; init; } = 3.5f;
    public uint LogScaleFloor { get; init; } = 4;
    public float LogScaleAlpha { get; init; } = 0.25f;
    public float ExpertWeightsScale { get; init; } = 2.0f;
    public float SituBeta { get; init; } = 2.0f;
    public float SituLinearBeta { get; init; } = 3.0f;
    public float Softcap { get; init; } = 8.0f;
    /// <summary>The GGUF's RMS epsilon; deliberately not the 1e-6 that mix and the loop controllers hard-code.</summary>
    public float Eps { get; init; } = 1e-5f;
    public uint ContextLength { get; init; } = 4096;

    /// <summary>Write <c>inkling2_loop</c> with the loop tensors and keys; false writes plain <c>inkling2</c>.</summary>
    public bool Looped { get; init; } = true;
    public int LoopStart { get; init; } = 2;
    public int LoopEnd { get; init; } = 6;
    /// <summary>Passes over the core; llama.cpp accepts 1 only with <see cref="LoopVarying"/>.</summary>
    public int LoopCount { get; init; } = 3;
    public int GateRank { get; init; } = 8;
    public bool LoopVarying { get; init; }

    public uint Bos { get; init; } = 1;
    public uint Eos { get; init; } = 2;

    public string Architecture => Looped ? "inkling2_loop" : "inkling2";
    public int Layers => Layout.Length;
    public bool IsSliding(int l) => Layout[l] == 's';
    public bool IsLoopCore(int l) => Looped && l >= LoopStart && l < LoopEnd;
    /// <summary>attn_rel_bias columns of layer <paramref name="l"/>.</summary>
    public int Extent(int l) => IsSliding(l) ? Window : RelativeExtent;
    /// <summary>Cache slots llama.cpp allocates, one per layer application: the physical layers, then (count - 1) copies
    /// of the core.</summary>
    public int Slots => Layers + (Looped ? (LoopCount - 1) * (LoopEnd - LoopStart) : 0);

    /// <summary>0: every tensor's values are seeded by its name (the fixture every test so far used); otherwise by its name
    /// salted with the seed, for a sample of fixtures of the same shape.</summary>
    public int Seed { get; init; }

    /// <summary>exp_probs_b's scale: -1 negates the selection bias, a routing slip the route checks must catch.</summary>
    public float ExpertBiasScale { get; init; } = 1f;

    /// <summary>Every tensor with its F32 values, as <see cref="Write"/> stores them.</summary>
    public List<Tensor> Tensors()
    {
        Validate();
        int H = Hidden, d = HeadDim, taps = ConvTaps, rel = RelativeLength;
        var t = new List<Tensor>
        {
            G("token_embd.weight", 1.0f, H, VocabPadded),
            GA("token_embd_norm.weight", 1f, 0.1f, H),
            G("output.weight", 1.0f, H, VocabPadded),
            GA("output_norm.weight", 1f, 0.1f, H),
            G("output_res_query.weight", 2.0f, H),
        };
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            int qd = Heads * d, kvd = KvHeads * d;
            t.Add(G(p + "attn_q.weight", 0.25f, H, qd));
            t.Add(G(p + "attn_k.weight", 0.5f, H, kvd));
            t.Add(G(p + "attn_v.weight", 0.3f, H, kvd));
            t.Add(G(p + "attn_output.weight", 0.6f, qd, H));
            t.Add(G(p + "attn_gate.weight", 0.3f, H, qd));
            t.Add(GA(p + "attn_q_clip.weight", 1f, 0.3f, d, Heads));
            t.Add(G(p + "attn_sink_proj.weight", 0.3f, H, Heads));
            t.Add(G(p + "attn_rel.weight", 0.4f, H, rel * Heads));
            t.Add(G(p + "attn_rel_bias.weight", 1.0f, rel, Extent(l)));
            t.Add(GA(p + "attn_rel_clip.weight", 1f, 0.3f, rel, Heads));
            t.Add(G(p + "attn_k_conv.weight", 0.5f, taps, kvd));
            t.Add(G(p + "attn_v_conv.weight", 0.5f, taps, kvd));
            t.Add(G(p + "attn_post_conv.weight", 0.5f, taps, H));
            t.Add(G(p + "ffn_post_conv.weight", 0.5f, taps, H));
            if (IsSliding(l))
            {
                t.Add(GA(p + "attn_norm.weight", 1f, 0.1f, H));
            }
            else
            {
                t.Add(GA(p + "attn_q_inp_norm.weight", 1f, 0.1f, H));
                t.Add(GA(p + "attn_k_inp_norm.weight", 1f, 0.1f, H));
                t.Add(GA(p + "attn_v_inp_norm.weight", 1f, 0.1f, H));
                t.Add(G(p + "attn_k_conv_proj.weight", 0.1f, H, taps * KvHeads));
                t.Add(G(p + "attn_v_conv_proj.weight", 0.1f, H, taps * KvHeads));
            }
            if (l > 0)
                t.Add(IsSliding(l) ? G(p + "attn_res_query.weight", 2.0f, H) : G(p + "attn_res_query.weight", 2.0f, H, 3));
            t.Add(G(p + "ffn_res_query.weight", 2.0f, H));
            t.Add(GA(p + "ffn_norm.weight", 1f, 0.1f, H));
            t.Add(G(p + "ffn_gate_inp.weight", 0.3f, H, Experts));
            t.Add(G(p + "exp_probs_b.bias", 0.15f * ExpertBiasScale, Experts));
            t.Add(G(p + "ffn_gate_up_exps.weight", 0.3f, H, 2 * ExpertFf, Experts));
            t.Add(G(p + "ffn_down_exps.weight", 0.3f, ExpertFf, H, Experts));
            if (IsLoopCore(l))
            {
                t.Add(G(p + "loop_msg_query.weight", 2.0f, H, 4));
                foreach (string c in new[] { "q", "k", "v", "ffn" })
                {
                    t.Add(G(p + $"loop_{c}_ctrl_down.weight", 0.15f, 2 * H, GateRank));
                    t.Add(G(p + $"loop_{c}_ctrl_gain.weight", 0.5f, GateRank, H));
                    t.Add(G(p + $"loop_{c}_ctrl_forget.weight", 0.5f, GateRank, H));
                }
            }
        }
        return t;

        // Seed 0: the values come from the names alone; another seed salts each name (the tensor keeps its own).
        Tensor G(string name, float scale, params int[] dims) => Named(Gen(Salted(name), scale, dims), name);
        Tensor GA(string name, float center, float scale, params int[] dims) => Named(GenAround(Salted(name), center, scale, dims), name);
        string Salted(string name) => Seed == 0 ? name : $"{name}#seed{Seed}";
        static Tensor Named(Tensor tensor, string name)
        {
            tensor.Name = name;
            return tensor;
        }
    }

    /// <summary>The metadata, keyed as convert_hf_to_gguf writes the real checkpoint (prefix = the architecture).</summary>
    public List<Kv> Metadata()
    {
        Validate();
        string a = Architecture;
        var heads = new int[Layers];
        var kvHeads = new int[Layers];
        var pattern = new bool[Layers];
        for (int l = 0; l < Layers; l++)
        {
            heads[l] = Heads;
            kvHeads[l] = KvHeads;
            pattern[l] = IsSliding(l);
        }
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = ContextLength },
            new U32 { Key = $"{a}.embedding_length", V = (uint)Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = (uint)ExpertFf },
            new U32 { Key = $"{a}.expert_feed_forward_length", V = (uint)ExpertFf },
            new I32Arr { Key = $"{a}.attention.head_count", V = heads },
            new I32Arr { Key = $"{a}.attention.head_count_kv", V = kvHeads },
            new U32 { Key = $"{a}.attention.key_length", V = (uint)HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = (uint)HeadDim },
            new U32 { Key = $"{a}.attention.key_length_swa", V = (uint)HeadDim },
            new U32 { Key = $"{a}.attention.value_length_swa", V = (uint)HeadDim },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = Eps },
            new U32 { Key = $"{a}.expert_count", V = (uint)Experts },
            new U32 { Key = $"{a}.expert_used_count", V = (uint)ExpertsUsed },
            new U32 { Key = $"{a}.vocab_size", V = (uint)VocabPadded },
            new U32 { Key = $"{a}.vocab_size_unpadded", V = (uint)Vocab },
            new U32 { Key = $"{a}.rope.dimension_count", V = 0 },
            new U32 { Key = $"{a}.attention.sliding_window", V = (uint)Window },
            new BoolArr { Key = $"{a}.attention.sliding_window_pattern", V = pattern },
            new U32 { Key = $"{a}.shortconv.l_cache", V = (uint)ConvTaps },
            new F32 { Key = $"{a}.expert_weights_scale", V = ExpertWeightsScale },
            new F32 { Key = $"{a}.activation.situ_beta", V = SituBeta },
            new F32 { Key = $"{a}.activation.situ_linear_beta", V = SituLinearBeta },
            new U32 { Key = $"{a}.attn_res.block_size", V = (uint)BlockSize },
            new F32 { Key = $"{a}.final_logit_softcapping", V = Softcap },
            new U32 { Key = $"{a}.attention.relative_length", V = (uint)RelativeLength },
            new U32 { Key = $"{a}.attention.relative_extent", V = (uint)RelativeExtent },
            new F32 { Key = $"{a}.attention.qk_clip", V = QkClip },
            new F32 { Key = $"{a}.attention.relative_clip", V = RelativeClip },
            new U32 { Key = $"{a}.attention.log_scale_floor", V = LogScaleFloor },
            new F32 { Key = $"{a}.attention.log_scale_alpha", V = LogScaleAlpha },
        };
        if (Looped)
        {
            kv.Add(new U32 { Key = $"{a}.loop.start", V = (uint)LoopStart });
            kv.Add(new U32 { Key = $"{a}.loop.end", V = (uint)LoopEnd });
            kv.Add(new U32 { Key = $"{a}.loop.count", V = (uint)LoopCount });
            kv.Add(new U32 { Key = $"{a}.loop.gate_rank", V = (uint)GateRank });
            kv.Add(new Bool { Key = $"{a}.loop.varying", V = LoopVarying });
        }
        AddByteTokenizer(kv, Vocab, Bos, Eos);
        ((Str)kv.Single(e => e.Key == "tokenizer.ggml.pre")).V = "inkling2";
        return kv;
    }

    /// <summary>The routed experts' stacks as F16, as the shared runtime's MoE fixtures store them (F32 three-dimensional
    /// tensors load unstacked); every other weight F32. The reference reads the same rounded values from the file.</summary>
    public bool HalfExperts { get; init; } = true;

    public string Write(string path)
    {
        var tensors = Tensors();
        if (HalfExperts)
            foreach (Tensor t in tensors)
                if (t.Dims.Length == 3 && t.Name.Contains("_exps.", StringComparison.Ordinal))
                {
                    t.Type = SyntheticGguf.GgmlType.F16;
                    t.Data = Array.ConvertAll(t.Data, v => (float)(Half)v);
                }
        SyntheticGguf.Write(path, Metadata(), tensors);
        return path;
    }

    /// <summary>llama.cpp's load-time checks (inkling2.cpp, inkling2-loop.cpp) that these dims could break.</summary>
    private void Validate()
    {
        if (Layout.Any(c => c != 's' && c != 'g'))
            throw new ArgumentException($"Layout '{Layout}' must be 's' (sliding) and 'g' (global) only.");
        if (Heads % KvHeads != 0 || ExpertsUsed > Experts || Vocab > VocabPadded || ConvTaps < 2 || BlockSize < 1)
            throw new ArgumentException("Invalid Inkling 2 attention, expert, vocabulary or conv dims.");
        if (Looped && (LoopCount < (LoopVarying ? 1 : 2) || GateRank < 1 || LoopStart <= 0 || LoopStart >= LoopEnd ||
                       LoopEnd > Layers || LoopStart % BlockSize != 0 || LoopEnd % BlockSize != 0))
            throw new ArgumentException(
                $"Invalid loop core [{LoopStart}, {LoopEnd}) with block size {BlockSize}, {LoopCount} passes (varying {LoopVarying}).");
    }
}
