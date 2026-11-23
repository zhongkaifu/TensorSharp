// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>
/// A small KRAFTON Raon-1 nano checkpoint (GGUF <c>raon1nano</c>) in F32, with the real file's tensor names, ne0-first shapes and
/// metadata keys (raon-1-nano-Q8_0.gguf; the converter's folded layout): four hyper-connection streams with their
/// [fn | base | chart] read and write sites, per-layer embeddings (a token table and a model projection), RDA layers (fused
/// q|k|v|g, a causal conv, low-rank decay with the b and b+w gates in ssm_f_a's tail rows) and absorbed, NoPE, gated MLA layers
/// (attn_k_b / attn_v_b per head). Weights depend only on their names (<see cref="SyntheticGguf.Gen"/>).
/// </summary>
internal sealed class RaonSyntheticModelBuilder
{
    public int Hidden { get; init; } = 32;
    public int Hc { get; init; } = 4;
    public int PleDim { get; init; } = 8;
    public int Ffn { get; init; } = 48;
    public int MlaHeads { get; init; } = 2;
    public int QLora { get; init; } = 16;
    public int KvLora { get; init; } = 16;
    public int Extra { get; init; } = 8;
    /// <summary>key_length_mla: the nope part (QkHead - Extra) plus the extra channels.</summary>
    public int QkHead { get; init; } = 16;
    public int VMla { get; init; } = 8;
    /// <summary>ggml-cuda's gated delta net runs head dims 16, 32, 64 and 128. With 4 heads of 32 the RDA state (16 KB) and
    /// conv window (3 x 384 floats) pass 4 KB: the GGML graph builder keeps a smaller state leaf in per-run compute memory, not
    /// in the device copy a checkpoint or slot copy on the device reads and writes (the real model's are 1 MB and 72 KB).</summary>
    public int RdaHeads { get; init; } = 4;
    public int RdaDim { get; init; } = 32;
    public int DecayRank { get; init; } = 8;
    public int ConvKernel { get; init; } = 4;
    /// <summary>Layer kinds: 'r' RDA, 'm' MLA.</summary>
    public string Layout { get; init; } = "rrmrrm";
    /// <summary>The shared slot, engine and checkpoint scripts use token ids up to 124.</summary>
    public int Vocab { get; init; } = 128;
    public float Eps { get; init; } = 1e-6f;
    public uint Eos { get; init; } = 2;

    public int Layers => Layout.Length;
    public bool IsRda(int l) => Layout[l] == 'r';
    private int Mix => 2 * Hc + (Hc - 1) * (Hc - 1);

    public List<Tensor> Tensors()
    {
        int H = Hidden, inner = RdaHeads * RdaDim, cells = Hc * Hc;
        var t = new List<Tensor>
        {
            G("token_embd.weight", 1.0f, H, Vocab),
            GA("output_norm.weight", 1f, 0.1f, H),
            G("per_layer_token_embd.weight", 0.3f, Layers * PleDim, Vocab),
            G("per_layer_model_proj.weight", 0.3f, H, Layers * PleDim),
            GA("per_layer_proj_norm.weight", 1f, 0.1f, PleDim),
        };
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            foreach (string site in new[] { "attn", "ffn" })
            {
                t.Add(G(p + $"hc_{site}_fn.weight", 0.5f, Hc * H, Mix));
                t.Add(G(p + $"hc_{site}_base.weight", 0.5f, Mix));
                // K (Mix x cells, row-major [cells][Mix]) | r (cells) | c0 (cells).
                t.Add(G(p + $"hc_{site}_chart.weight", 0.3f, Mix * cells + 2 * cells));
            }
            t.Add(GA(p + "attn_norm.weight", 1f, 0.1f, H));
            t.Add(GA(p + "ffn_norm.weight", 1f, 0.1f, H));
            t.Add(G(p + "ffn_up.weight", 0.3f, H, 2 * Ffn));
            t.Add(G(p + "ffn_down.weight", 0.3f, Ffn, H));
            t.Add(G(p + "inp_gate.weight", 0.3f, H, PleDim));
            t.Add(G(p + "proj.weight", 0.3f, PleDim, H));
            t.Add(GA(p + "post_norm.weight", 1f, 0.1f, H));
            if (IsRda(l))
            {
                t.Add(G(p + "attn_qkv.weight", 0.3f, H, 4 * inner));
                t.Add(G(p + "ssm_conv1d.weight", 0.5f, ConvKernel, 3 * inner));
                t.Add(G(p + "ssm_f_a.weight", 0.3f, H, DecayRank + 2 * RdaHeads));
                t.Add(G(p + "ssm_f_b.weight", 0.3f, DecayRank, inner));
                t.Add(G(p + "ssm_dt.bias", 0.5f, inner));
                t.Add(GA(p + "ssm_norm.weight", 1f, 0.1f, RdaDim));
                t.Add(G(p + "attn_output.weight", 0.3f, inner, H));
            }
            else
            {
                int nope = QkHead - Extra;
                t.Add(G(p + "attn_q_a.weight", 0.3f, H, QLora));
                t.Add(GA(p + "attn_q_a_norm.weight", 1f, 0.1f, QLora));
                t.Add(G(p + "attn_q_b.weight", 0.3f, QLora, MlaHeads * QkHead));
                t.Add(G(p + "attn_kv_a_mqa.weight", 0.3f, H, KvLora + Extra));
                t.Add(GA(p + "attn_kv_a_norm.weight", 1f, 0.1f, KvLora));
                t.Add(G(p + "attn_k_b.weight", 0.3f, nope, KvLora, MlaHeads));
                t.Add(G(p + "attn_v_b.weight", 0.3f, KvLora, VMla, MlaHeads));
                t.Add(G(p + "attn_gate.weight", 0.3f, H, MlaHeads * VMla));
                t.Add(G(p + "attn_output.weight", 0.3f, MlaHeads * VMla, H));
            }
        }
        return t;

        static Tensor G(string name, float scale, params int[] dims) => Named(Gen(name, scale, dims), name);
        static Tensor GA(string name, float center, float scale, params int[] dims) => Named(GenAround(name, center, scale, dims), name);
        static Tensor Named(Tensor tensor, string name)
        {
            tensor.Name = name;
            return tensor;
        }
    }

    public List<Kv> Metadata()
    {
        const string a = "raon1nano";
        var heads = new int[Layers];
        var kvHeads = new int[Layers];
        for (int l = 0; l < Layers; l++)
        {
            heads[l] = IsRda(l) ? RdaHeads : MlaHeads;
            kvHeads[l] = IsRda(l) ? 0 : 1;
        }
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = (uint)Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = (uint)Ffn },
            new I32Arr { Key = $"{a}.attention.head_count", V = heads },
            new I32Arr { Key = $"{a}.attention.head_count_kv", V = kvHeads },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = Eps },
            new U32 { Key = $"{a}.attention.key_length", V = (uint)(KvLora + Extra) },
            new U32 { Key = $"{a}.attention.value_length", V = (uint)KvLora },
            new U32 { Key = $"{a}.vocab_size", V = (uint)Vocab },
            new U32 { Key = $"{a}.ssm.conv_kernel", V = (uint)ConvKernel },
            new U32 { Key = $"{a}.ssm.time_step_rank", V = (uint)DecayRank },
            new U32 { Key = $"{a}.kda.head_dim", V = (uint)RdaDim },
            new F32 { Key = $"{a}.kda.gate_lower_bound", V = -5f },
            new U32 { Key = $"{a}.attention.q_lora_rank", V = (uint)QLora },
            new U32 { Key = $"{a}.attention.kv_lora_rank", V = (uint)KvLora },
            new U32 { Key = $"{a}.rope.dimension_count", V = (uint)Extra },
            new U32 { Key = $"{a}.attention.key_length_mla", V = (uint)QkHead },
            new U32 { Key = $"{a}.attention.value_length_mla", V = (uint)VMla },
            new F32 { Key = $"{a}.attention.latent_norm_rms_epsilon", V = Eps },
            new U32 { Key = $"{a}.embedding_length_per_layer_input", V = (uint)PleDim },
            new U32 { Key = $"{a}.hyper_connection.count", V = (uint)Hc },
            new F32 { Key = $"{a}.hyper_connection.epsilon", V = 1e-5f },
            new F32 { Key = $"{a}.hyper_connection.magnitude", V = 1f },
        };
        AddByteTokenizer(kv, Vocab, Eos, Eos);
        return kv;
    }

    public string Write(string path)
    {
        SyntheticGguf.Write(path, Metadata(), Tensors());
        return path;
    }
}
