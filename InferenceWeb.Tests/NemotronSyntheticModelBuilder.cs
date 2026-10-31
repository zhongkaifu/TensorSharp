// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Builds a tiny, deterministic nemotron_h_moe GGUF: attention and MoE FFN layers,
// Q8_0 projections and routed experts, an F32 router with an exp_probs_b selection
// bias, optional shared experts and latent MoE, and - optionally - llama.cpp-style
// NVFP4 ".scale" sidecars. A layout may also hold Mamba2 layers and, with no MoE
// layer, dense ReLU^2 FFN layers (then the file is a dense nemotron_h hybrid).
//
// The sidecar file and its twin hold the SAME effective weights W: the twin stores
// Q8_0(W) with no sidecars, the scaled file stores Q8_0(W / s) plus "<base>.scale"
// = s. Every scale is a power of two (or -1), so Q8_0 is exactly equivariant -
// Q8_0(W / s) * s == Q8_0(W) bit for bit while the F16 block scales stay normal -
// and a model that applies every scale at the right place computes what the twin
// computes, up to float rounding. A missing scale is off by its factor; an up scale
// applied after ReLU^2 instead of before is off by its square (and by its sign for
// the -1 expert, which the router bias pins into every token's top-k).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace InferenceWeb.Tests;

internal sealed class NemotronSyntheticModelBuilder
{
    public const int Hidden = 64;
    public const int NumHeads = 4;
    public const int NumKvHeads = 2;
    public const int HeadDim = 16;
    public const int Vocab = 256;
    public const int SharedFf = 64;
    public const int LatentDim = 32;
    public const int DenseFf = 64;

    // Mamba2 geometry: 4 heads of 16, 2 groups, state 128 (as Nemotron-H; ggml-cuda's
    // ssm_scan, behind the native prefill, takes only 96, 128 or 256), conv kernel 4.
    public const int SsmInner = 64;
    public const int SsmState = 128;
    public const int SsmHeads = 4;
    public const int SsmGroups = 2;
    public const int SsmConv = 4;
    /// <summary>ssm_in's output width: z, x, B, C and dt.</summary>
    public const int SsmInProj = 2 * SsmInner + 2 * SsmGroups * SsmState + SsmHeads;

    /// <summary>Per-expert up scales, cycled over the experts. Expert 5 (-1) flips
    /// the sign of its up projection, which ReLU^2 only survives if the scale is
    /// applied BEFORE the activation.</summary>
    public static readonly float[] UpScaleCycle = { 1f / 1024f, 1f / 16f, 0.5f, 2f, 4f, -1f, 1f, 0.25f };

    /// <summary>Per-expert down scales, cycled over the experts.</summary>
    public static readonly float[] DownScaleCycle = { 0.5f, 2f, 4f, 1f / 16f, 0.25f, 2f, 1f, 8f };

    /// <summary>The expert the router bias pins into every token's top-k.</summary>
    public const int PinnedExpert = 5;

    /// <summary>Layer kinds in order: 'M' = MoE FFN layer, 'A' = attention layer,
    /// 'S' = Mamba2 layer, 'F' = dense ReLU^2 FFN layer (not with 'M': the model runs every
    /// FFN layer as MoE once the file has experts).</summary>
    public string Layout { get; init; } = "MAM";

    /// <summary>Attention geometry of the 'A' layers (defaults: the constants above). The
    /// direct-CUDA prefill tests use the real models' head dim 128 at GQA group 16
    /// (Nemotron-3.5 Lightning), which picks a different attention route than head dim 16.</summary>
    public int AttentionHeads { get; init; } = NumHeads;
    public int AttentionKvHeads { get; init; } = NumKvHeads;
    public int AttentionHeadDim { get; init; } = HeadDim;

    /// <summary>nemotron_h_moe with MoE layers, the dense nemotron_h otherwise.</summary>
    public string Architecture => Layout.Contains('M') ? "nemotron_h_moe" : "nemotron_h";

    /// <summary>Storage type of the projections (attention, shared and latent experts,
    /// Mamba2 in/out, dense FFN, LM head). Q8_0 (default), F16 and BF16 are quantized from the
    /// name-seeded values; a K/IQ/MXFP4 block type is written as seeded block bytes
    /// (SyntheticGguf.Blocks), which needs every input width to be a multiple of its block (256
    /// for Q4_K: use <see cref="Width"/> 256) and works neither with <see cref="Scaled"/> nor
    /// with <see cref="Transform"/>.</summary>
    public SyntheticGguf.GgmlType LinearType { get; init; } = SyntheticGguf.GgmlType.Q8_0;

    /// <summary>Storage type of the routed experts' up (and, unless <see cref="DownExpertType"/>
    /// says otherwise, down) projections: Q8_0 by default; MXFP4 / Q5_1 / K types as seeded
    /// blocks, as Nemotron-3.5 Lightning stores MXFP4 up and Q5_1 down experts.</summary>
    public SyntheticGguf.GgmlType ExpertType { get; init; } = SyntheticGguf.GgmlType.Q8_0;
    public SyntheticGguf.GgmlType? DownExpertType { get; init; }

    /// <summary>Model and FFN widths (defaults: the constants above). The real models' K is a
    /// multiple of 256, so their K-quant routes need <see cref="Width"/> 256 here.</summary>
    public int Width { get; init; } = Hidden;
    public int SsmInnerWidth { get; init; } = SsmInner;
    public int DenseFfWidth { get; init; } = DenseFf;
    public int SharedFfWidth { get; init; } = SharedFf;

    /// <summary>ssm_in's output width for <see cref="SsmInnerWidth"/>: z, x, B, C and dt.</summary>
    private int SsmInProjWidth => 2 * SsmInnerWidth + 2 * SsmGroups * SsmState + SsmHeads;

    private static bool IsBlockType(SyntheticGguf.GgmlType t)
        => t is not (SyntheticGguf.GgmlType.F32 or SyntheticGguf.GgmlType.F16 or SyntheticGguf.GgmlType.BF16 or SyntheticGguf.GgmlType.Q8_0);

    /// <summary>
    /// Rewrite tensors before they are written (a LoRA test's merged-weights twin): called
    /// with each tensor's GGUF name and its values as the model will read them (an F16 or Q8_0
    /// tensor's already rounded), it returns the values to write, or null to keep the tensor.
    /// A rewritten tensor is stored as F32. Not with <see cref="Scaled"/>, whose stored values
    /// are not the effective weights.
    /// </summary>
    public Func<string, float[], float[]> Transform { get; init; }

    /// <summary>Routed experts per MoE layer (all layers unless <see cref="LayerExperts"/>).</summary>
    public int Experts { get; init; } = 8;

    /// <summary>Optional per-MoE-layer expert counts (Puzzle/Bolt-style varied layout).</summary>
    public int[] LayerExperts { get; init; }

    public int ExpertFf { get; init; } = 64;
    public int TopK { get; init; } = 3;

    /// <summary>When set, expert_used_count is written as this per-layer array (and the
    /// scalar reads as its element [0]); otherwise as the scalar <see cref="TopK"/>.</summary>
    public uint[] UsedPerLayer { get; init; }

    public bool Latent { get; init; }

    /// <summary>Width of the latent MoE space (with <see cref="Latent"/>).</summary>
    public int LatentWidth { get; init; } = LatentDim;

    public bool SharedExperts { get; init; } = true;

    /// <summary>Write raw = W / s plus sidecars (true) or the twin Q8_0(W) (false).</summary>
    public bool Scaled { get; init; }

    /// <summary>attn_q/k/v sidecar scales (equal ones fuse, unequal ones stay separate).</summary>
    public float[] QkvScales { get; init; } = { 0.5f, 0.5f, 0.5f };
    public float AttnOutputScale { get; init; } = 2f;
    public float SharedUpScale { get; init; } = 4f;
    public float SharedDownScale { get; init; } = 0.25f;
    public float LatentInScale { get; init; } = 0.5f;
    public float LatentOutScale { get; init; } = 2f;
    public bool ExpertUpScales { get; init; } = true;
    public bool ExpertDownScales { get; init; } = true;

    /// <summary>Store attn_v as F32 instead of Q8_0, so Q/K/V cannot fuse by type.</summary>
    public bool F32AttnV { get; init; }

    private int ExpertsOfLayer(int moeOrdinal) => LayerExperts != null ? LayerExperts[moeOrdinal] : Experts;

    public string Write(string path)
    {
        bool moeLayers = Layout.Contains('M');
        if (moeLayers && Layout.Contains('F'))
            throw new ArgumentException("A layout holds MoE ('M') or dense ('F') FFN layers, not both.");
        if (Scaled && Transform != null)
            throw new ArgumentException("Transform rewrites effective weights, which a Scaled file does not store.");

        var kv = new List<SyntheticGguf.Kv>();
        var tensors = new List<SyntheticGguf.Tensor>();
        int layers = Layout.Length;
        var headCountKv = new uint[layers];
        var ffnLength = new uint[layers];
        for (int l = 0; l < layers; l++)
        {
            // The model reads the layer kind from these two arrays: attention has KV heads,
            // an FFN layer a width, a Mamba2 layer neither.
            char kind = Layout[l];
            headCountKv[l] = kind == 'A' ? (uint)AttentionKvHeads : 0u;
            ffnLength[l] = kind switch
            {
                'M' => (uint)ExpertFf,
                'F' => (uint)DenseFfWidth,
                'A' or 'S' => 0u,
                _ => throw new ArgumentException($"Unknown layer kind '{kind}' in layout {Layout}."),
            };
        }

        string a = Architecture;
        kv.Add(new SyntheticGguf.Str { Key = "general.architecture", V = a });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.block_count", V = (uint)layers });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.context_length", V = 256 });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.embedding_length", V = (uint)Width });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.attention.head_count", V = (uint)AttentionHeads });
        kv.Add(new SyntheticGguf.U32Arr { Key = $"{a}.attention.head_count_kv", V = headCountKv });
        kv.Add(new SyntheticGguf.U32Arr { Key = $"{a}.feed_forward_length", V = ffnLength });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.attention.key_length", V = (uint)AttentionHeadDim });
        kv.Add(new SyntheticGguf.U32 { Key = $"{a}.attention.value_length", V = (uint)AttentionHeadDim });
        kv.Add(new SyntheticGguf.F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-5f });
        if (moeLayers)
        {
            int maxExperts = LayerExperts != null ? LayerExperts.Max() : Experts;
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.expert_count", V = (uint)maxExperts });
            if (UsedPerLayer != null)
                kv.Add(new SyntheticGguf.U32Arr { Key = $"{a}.expert_used_count", V = UsedPerLayer });
            else
                kv.Add(new SyntheticGguf.U32 { Key = $"{a}.expert_used_count", V = (uint)TopK });
            kv.Add(new SyntheticGguf.Bool { Key = $"{a}.expert_weights_norm", V = true });
            kv.Add(new SyntheticGguf.F32 { Key = $"{a}.expert_weights_scale", V = 2f });
        }
        if (Layout.Contains('S'))
        {
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.ssm.conv_kernel", V = SsmConv });
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.ssm.inner_size", V = (uint)SsmInnerWidth });
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.ssm.state_size", V = SsmState });
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.ssm.time_step_rank", V = SsmHeads });
            kv.Add(new SyntheticGguf.U32 { Key = $"{a}.ssm.group_count", V = SsmGroups });
        }
        SyntheticGguf.AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        tensors.Add(SyntheticGguf.Gen("token_embd.weight", 0.5f, Width, Vocab));
        tensors.Add(SyntheticGguf.GenAround("output_norm.weight", 1f, 0.1f, Width));
        tensors.Add(IsBlockType(LinearType)
            ? SyntheticGguf.Blocks("output.weight", LinearType, 0.1f, Width, Vocab)
            : Linear(SyntheticGguf.Gen("output.weight", 0.1f, Width, Vocab)));

        int moeOrdinal = 0;
        for (int l = 0; l < layers; l++)
        {
            string p = $"blk.{l}.";
            tensors.Add(SyntheticGguf.GenAround(p + "attn_norm.weight", 1f, 0.1f, Width));
            if (Layout[l] == 'A')
            {
                int qDim = AttentionHeads * AttentionHeadDim, kvDim = AttentionKvHeads * AttentionHeadDim;
                AddLinear(tensors, p + "attn_q", 0.15f, QkvScales[0], Width, qDim);
                AddLinear(tensors, p + "attn_k", 0.15f, QkvScales[1], Width, kvDim);
                if (F32AttnV)
                    tensors.Add(SyntheticGguf.Gen(p + "attn_v.weight", 0.15f, Width, kvDim));
                else
                    AddLinear(tensors, p + "attn_v", 0.15f, QkvScales[2], Width, kvDim);
                AddLinear(tensors, p + "attn_output", 0.15f, AttnOutputScale, qDim, Width);
                continue;
            }
            if (Layout[l] == 'S')
            {
                AddMamba2(tensors, p);
                continue;
            }
            if (Layout[l] == 'F')
            {
                AddLinear(tensors, p + "ffn_up", 0.2f, 1f, Width, DenseFfWidth);
                AddLinear(tensors, p + "ffn_down", 0.2f, 1f, DenseFfWidth, Width);
                continue;
            }

            int experts = ExpertsOfLayer(moeOrdinal++);
            int expertIn = Latent ? LatentWidth : Width;
            tensors.Add(SyntheticGguf.Gen(p + "ffn_gate_inp.weight", 0.3f, Width, experts));
            var bias = new float[experts];
            if (experts > PinnedExpert)
                bias[PinnedExpert] = 8f;
            tensors.Add(new SyntheticGguf.Tensor { Name = p + "exp_probs_b.bias", Dims = new[] { (ulong)experts }, Data = bias });

            if (Latent)
            {
                AddLinear(tensors, p + "ffn_latent_down", 0.2f, LatentInScale, Width, LatentWidth);
                AddLinear(tensors, p + "ffn_latent_up", 0.2f, LatentOutScale, LatentWidth, Width);
            }
            if (SharedExperts)
            {
                AddLinear(tensors, p + "ffn_up_shexp", 0.2f, SharedUpScale, Width, SharedFfWidth);
                AddLinear(tensors, p + "ffn_down_shexp", 0.2f, SharedDownScale, SharedFfWidth, Width);
            }

            AddExperts(tensors, p + "ffn_up_exps", 0.25f, ExpertUpScales ? UpScaleCycle : null, expertIn, ExpertFf, experts);
            AddExperts(tensors, p + "ffn_down_exps", 0.25f, ExpertDownScales ? DownScaleCycle : null, ExpertFf, expertIn, experts);
        }

        if (Transform != null)
        {
            for (int i = 0; i < tensors.Count; i++)
            {
                var t = tensors[i];
                float[] rewritten = Transform(t.Name, Decoded(t));
                if (rewritten != null)
                    tensors[i] = new SyntheticGguf.Tensor { Name = t.Name, Dims = t.Dims, Data = rewritten };
            }
        }

        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    private static SyntheticGguf.Tensor Q8(SyntheticGguf.Tensor t)
    {
        t.Type = SyntheticGguf.GgmlType.Q8_0;
        return t;
    }

    private SyntheticGguf.Tensor Linear(SyntheticGguf.Tensor t)
    {
        t.Type = LinearType;
        return t;
    }

    /// <summary>A Mamba2 layer as llama.cpp's converter writes one: ssm_in and ssm_out
    /// projections, an F32 depthwise conv (kernel-fastest), dt bias, A = -exp(A_log)
    /// (negative, so the state decays), D and the grouped RMSNorm weight.</summary>
    private void AddMamba2(List<SyntheticGguf.Tensor> tensors, string p)
    {
        int convChannels = SsmInnerWidth + 2 * SsmGroups * SsmState;
        AddLinear(tensors, p + "ssm_in", 0.15f, 1f, Width, SsmInProjWidth);
        tensors.Add(SyntheticGguf.Gen(p + "ssm_conv1d.weight", 0.3f, SsmConv, convChannels));
        tensors.Add(SyntheticGguf.Gen(p + "ssm_conv1d.bias", 0.1f, convChannels));
        tensors.Add(SyntheticGguf.Gen(p + "ssm_dt.bias", 0.5f, SsmHeads));
        tensors.Add(SyntheticGguf.GenAround(p + "ssm_a", -1f, 0.25f, 1, SsmHeads));
        tensors.Add(SyntheticGguf.GenAround(p + "ssm_d", 1f, 0.1f, 1, SsmHeads));
        tensors.Add(SyntheticGguf.GenAround(p + "ssm_norm.weight", 1f, 0.1f, SsmInnerWidth / SsmGroups, SsmGroups));
        AddLinear(tensors, p + "ssm_out", 0.15f, 1f, SsmInnerWidth, Width);
    }

    /// <summary>The values a model reads back from <paramref name="t"/> as stored.</summary>
    private static float[] Decoded(SyntheticGguf.Tensor t)
    {
        switch (t.Type)
        {
            case SyntheticGguf.GgmlType.F32:
                return (float[])t.Data.Clone();
            case SyntheticGguf.GgmlType.F16:
                return Array.ConvertAll(t.Data, v => (float)(Half)v);
            case SyntheticGguf.GgmlType.Q8_0:
            {
                // Block: F16 scale d, then 32 int8 codes q; value = d * q.
                byte[] raw = SyntheticGguf.QuantizeQ8_0(t.Data, (int)t.Dims[0]);
                var values = new float[t.Data.Length];
                for (int b = 0; b < values.Length / 32; b++)
                {
                    float d = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(raw, b * 34));
                    for (int j = 0; j < 32; j++)
                        values[b * 32 + j] = d * (sbyte)raw[b * 34 + 2 + j];
                }
                return values;
            }
            default:
                throw new NotSupportedException($"{t.Name}: no decoder for {t.Type}.");
        }
    }

    /// <summary>A projection [inDim -> outDim] stored as <see cref="LinearType"/> whose effective
    /// weight is the name-seeded W; with sidecars the file stores W / s and "<paramref name="baseName"/>.scale" = s.</summary>
    private void AddLinear(List<SyntheticGguf.Tensor> tensors, string baseName, float magnitude, float scale, int inDim, int outDim)
    {
        if (IsBlockType(LinearType))
        {
            if (Scaled || Transform != null)
                throw new NotSupportedException($"{LinearType} projections are seeded blocks: no sidecar scales, no Transform.");
            tensors.Add(SyntheticGguf.Blocks(baseName + ".weight", LinearType, magnitude, inDim, outDim));
            return;
        }
        var t = Linear(SyntheticGguf.Gen(baseName + ".weight", magnitude, inDim, outDim));
        if (Scaled && scale != 1f)
        {
            for (int i = 0; i < t.Data.Length; i++)
                t.Data[i] /= scale;
            tensors.Add(new SyntheticGguf.Tensor { Name = baseName + ".scale", Dims = new[] { 1UL }, Data = new[] { scale } });
            // NVFP4 GGUFs also carry calibration ".input_scale" tensors, which llama.cpp ignores.
            tensors.Add(new SyntheticGguf.Tensor { Name = baseName + ".input_scale", Dims = new[] { 1UL }, Data = new[] { 3f } });
        }
        tensors.Add(t);
    }

    /// <summary>A stacked [inDim, outDim, experts] Q8_0 expert tensor; with sidecars each
    /// expert's slice is divided by its own scale and "<paramref name="baseName"/>.scale"
    /// holds the [experts] vector.</summary>
    private void AddExperts(List<SyntheticGguf.Tensor> tensors, string baseName, float magnitude, float[] cycle,
        int inDim, int outDim, int experts)
    {
        var type = baseName.EndsWith("ffn_down_exps", StringComparison.Ordinal) ? DownExpertType ?? ExpertType : ExpertType;
        if (type != SyntheticGguf.GgmlType.Q8_0)
        {
            if (Scaled || Transform != null)
                throw new NotSupportedException($"{type} experts are seeded blocks: no sidecar scales, no Transform.");
            tensors.Add(SyntheticGguf.Blocks(baseName + ".weight", type, magnitude, inDim, outDim, experts));
            return;
        }
        var t = Q8(SyntheticGguf.Gen(baseName + ".weight", magnitude, inDim, outDim, experts));
        if (Scaled && cycle != null)
        {
            var scales = new float[experts];
            int perExpert = inDim * outDim;
            for (int e = 0; e < experts; e++)
            {
                scales[e] = cycle[e % cycle.Length];
                for (int i = 0; i < perExpert; i++)
                    t.Data[e * perExpert + i] /= scales[e];
            }
            tensors.Add(new SyntheticGguf.Tensor { Name = baseName + ".scale", Dims = new[] { (ulong)experts }, Data = scales });
        }
        tensors.Add(t);
    }
}
