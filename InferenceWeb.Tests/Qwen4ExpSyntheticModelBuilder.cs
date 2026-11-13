// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A tiny qwen4exp (Qwen3.8-Flash-Next) checkpoint with the real file's layout: the same tensor
// names, shapes and storage types (Q8_0 dense projections, F32 norms/router/decays, BF16 indexer
// projections, IQ3_S / IQ4_XS gate+up and IQ4_NL / Q8_0 down experts, a Q6_K head and an IQ4_NL
// n-gram table), shrunk to a 256-wide model. Layer 1 carries the PLE block, every 4th layer is
// full attention with QSA at ratio 4, and the indexer keeps 16 cells, so a prompt of a few dozen
// tokens already runs the sparse selection the real model reaches past 2048.
//
// It is an engine fixture, not a language model: the weights are seeded noise of a size that keeps
// every activation well conditioned. Both engines under comparison read the same bytes.
using System.Collections.Generic;
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

internal static class Qwen4ExpSyntheticModelBuilder
{
    public const int Layers = 8;
    public const int Hidden = 256;
    public const int Vocab = 256;
    public const int Hc = 4;
    public const int HcLowRank = 64;
    public const int Heads = 4;
    public const int KvHeads = 2;
    public const int HeadDim = 256;
    public const int RopeDims = 64;
    public const int Experts = 16;
    public const int ExpertsUsed = 4;
    public const int ExpertFf = 128;
    public const int SharedFf = 128;
    public const int ConvKernel = 4;
    public const int StateSize = 128;
    public const int KHeads = 2;
    public const int VHeads = 4;
    public const int IndexerHeads = 2;
    public const int IndexerDim = 128;
    public const int CompressRatio = 4;
    public const int PleNgram = 3;
    public const int PleHeadsPerNgram = 4;
    public const int PleHeadDim = 32;
    public const int PleLayer = 1;
    public const int EosToken = 255;
    public const int BosToken = 254;

    /// <summary>The layer types: GDN everywhere but every 4th layer.</summary>
    public static bool IsRecurrent(int il) => (il + 1) % 4 != 0;

    private static readonly ulong[] PleVocab = { 101, 103, 107, 109, 113, 127, 131, 137 };

    /// <summary>Write the checkpoint to <paramref name="path"/>; returns the path.
    /// <paramref name="q2kxlExperts"/> stores the experts as the UD-Q2_K_XL file does instead:
    /// IQ2_XS gate/up (IQ3_XXS in layer 2) over IQ4_NL down. Convolution types and
    /// <paramref name="matrixNormType"/> exercise converter differences for coefficients
    /// consumed as F32; a matrix norm preserves the ordinary flat scale values.
    /// Optional expert counts exercise router reductions at the trained model's width.
    /// <paramref name="q4kxlExperts"/> stores them as the UD-Q4_K_XL file does: Q4_K gate/up over Q5_1 down;
    /// <paramref name="expertGateUpType"/> and <paramref name="expertDownType"/> then override every layer's
    /// gate/up and down projections, and
    /// <paramref name="expertDownGain"/> scales the routed down projection's update (a quarter by default,
    /// as every block's output projection). <paramref name="expertFf"/> widens the experts (a multiple of
    /// 256 lets the down projection take a 256-value block format).</summary>
    public static string Write(string path, int indexerTopK = 16, int contextLength = 4096, bool q2kxlExperts = false,
        GgmlType pleConvType = GgmlType.F32, GgmlType ssmConvType = GgmlType.F32,
        GgmlType? matrixNormType = null, bool qsa = true, int expertCount = Experts, int expertUsedCount = ExpertsUsed,
        int hiddenSize = Hidden, bool q4kxlExperts = false, bool tiedOutput = false, GgmlType? expertDownType = null,
        float expertDownGain = 0.25f, GgmlType? expertGateUpType = null, int expertFf = ExpertFf, GgmlType routerType = GgmlType.F32,
        GgmlType projectionType = GgmlType.Q8_0)
    {
        if (expertCount < 1 || expertUsedCount < 1 || expertUsedCount > expertCount)
            throw new System.ArgumentOutOfRangeException(nameof(expertUsedCount), "Expert counts must be positive, and selected experts cannot exceed total experts.");
        if (hiddenSize < 256 || hiddenSize % 256 != 0)
            throw new System.ArgumentOutOfRangeException(nameof(hiddenSize), "Hidden size must contain complete 256-value quantization blocks.");
        int hcDim = Hc * hiddenSize;
        int pleHeadDim = hiddenSize / ((PleNgram - 1) * PleHeadsPerNgram);
        const int keyDim = StateSize * KHeads, valueDim = StateSize * VHeads, convDim = 2 * keyDim + valueDim;
        // Block-quantized rows must contain a whole block; the ordinary floating-point
        // fixtures retain the shipped four-tap convolution layout.
        int pleConvKernel = pleConvType == GgmlType.Q8_0 ? 32 : ConvKernel;
        int ssmConvKernel = ssmConvType == GgmlType.Q8_0 ? 32 : ConvKernel;
        var t = new List<Tensor>();

        // A projection y[out] = W[out, in] x[in] of unit-sized outputs for unit-sized inputs; a block's
        // output projection writes a small update into the residual (gain 0.25), as a trained model's
        // do, so rounding differences between engines stay the size they start at instead of
        // compounding through a chaotic stack.
        Tensor Proj(string name, int inDim, int outDim, GgmlType? explicitType = null, float gain = 1f)
        {
            GgmlType type = explicitType ?? projectionType;
            float s = gain * 1.5f / System.MathF.Sqrt(inDim);
            if (type is GgmlType.Q8_0 or GgmlType.F32 or GgmlType.BF16)
            {
                var w = Gen(name, s, inDim, outDim);
                w.Type = type;
                return w;
            }
            return Blocks(name, type, gain * 0.8f / System.MathF.Sqrt(inDim), inDim, outDim);
        }
        Tensor Experts3(string name, int inDim, int outDim, GgmlType type, float gain = 1f)
        {
            if (type is GgmlType.Q8_0 or GgmlType.BF16)
            {
                var w = Gen(name, gain * 1.5f / System.MathF.Sqrt(inDim), inDim, outDim, expertCount);
                w.Type = type;
                return w;
            }
            return Blocks(name, type, gain * 0.8f / System.MathF.Sqrt(inDim), inDim, outDim, expertCount);
        }
        const float OutGain = 0.25f;
        Tensor Norm(string name, int n)
        {
            Tensor norm = GenAround(name, 1f, 0.1f, n);
            if (matrixNormType is { } type)
            {
                norm.Type = type;
                norm.Dims = n == hcDim ? new ulong[] { (ulong)hiddenSize, Hc } : new ulong[] { (ulong)n, 1 };
            }
            return norm;
        }
        Tensor Conv(string name, GgmlType type, int kernel, int channels)
        {
            Tensor conv = Gen(name, 0.5f, kernel, channels);
            conv.Type = type;
            return conv;
        }

        var embd = Gen("token_embd.weight", 1.0f, hiddenSize, Vocab);
        embd.Type = GgmlType.Q8_0;
        t.Add(embd);
        if (!tiedOutput)
            t.Add(Blocks("output.weight", GgmlType.Q6_K, 1.2f / System.MathF.Sqrt(hiddenSize), hiddenSize, Vocab));
        t.Add(Norm("output_hc_norm.weight", hcDim));
        t.Add(Proj("output_hc_down.weight", hcDim, HcLowRank));
        t.Add(Proj("output_hc_up.weight", HcLowRank, hcDim));

        ulong pleRows = 0;
        foreach (ulong v in PleVocab) pleRows += v;
        t.Add(Blocks("per_layer_token_embd.weight", GgmlType.IQ4_NL, 0.5f, pleHeadDim, (int)pleRows));

        for (int il = 0; il < Layers; il++)
        {
            string p = $"blk.{il}.";
            foreach (string part in new[] { "attn", "ffn" })
            {
                t.Add(Norm(p + $"hc_{part}_norm.weight", hcDim));
                t.Add(Proj(p + $"hc_{part}_down.weight", hcDim, HcLowRank));
                t.Add(Proj(p + $"hc_{part}_up.weight", HcLowRank, hcDim));
                t.Add(Gen(p + $"hc_{part}_inject.weight", 1.5f / System.MathF.Sqrt(hcDim), hcDim, Hc));
            }

            if (IsRecurrent(il))
            {
                t.Add(Proj(p + "attn_qkv.weight", hiddenSize, convDim));
                t.Add(Proj(p + "attn_gate.weight", hiddenSize, valueDim));
                t.Add(Proj(p + "ssm_alpha.weight", hiddenSize, VHeads, GgmlType.F32));
                t.Add(Proj(p + "ssm_beta.weight", hiddenSize, VHeads, GgmlType.F32));
                t.Add(Conv(p + "ssm_conv1d.weight", ssmConvType, ssmConvKernel, convDim));
                t.Add(Gen(p + "ssm_dt.bias", 0.5f, VHeads));
                // ssm_a ships pre-negated (-exp(A_log)): the recurrence decays.
                t.Add(GenAround(p + "ssm_a", -0.9f, 0.3f, VHeads));
                t.Add(Norm(p + "ssm_norm.weight", StateSize));
                t.Add(Proj(p + "ssm_out.weight", valueDim, hiddenSize, gain: OutGain));
            }
            else
            {
                t.Add(Proj(p + "attn_q.weight", hiddenSize, Heads * 2 * HeadDim));
                t.Add(Proj(p + "attn_k.weight", hiddenSize, KvHeads * HeadDim));
                t.Add(Proj(p + "attn_v.weight", hiddenSize, KvHeads * HeadDim));
                t.Add(Norm(p + "attn_q_norm.weight", HeadDim));
                t.Add(Norm(p + "attn_k_norm.weight", HeadDim));
                t.Add(Proj(p + "attn_output.weight", Heads * HeadDim, hiddenSize, gain: OutGain));
                t.Add(Proj(p + "indexer.k_proj.weight", hiddenSize, IndexerDim, GgmlType.BF16));
                t.Add(Proj(p + "indexer.q_proj.weight", hiddenSize, IndexerHeads * IndexerDim, GgmlType.BF16));
                t.Add(Norm(p + "indexer.k_norm.weight", IndexerDim));
                t.Add(Norm(p + "indexer.q_norm.weight", IndexerDim));
            }

            if (il == PleLayer)
            {
                t.Add(Conv(p + "ple_conv1d.weight", pleConvType, pleConvKernel, hcDim));
                t.Add(Proj(p + "ple_key.weight", hiddenSize, hcDim));
                t.Add(Proj(p + "ple_value.weight", hiddenSize, hiddenSize, gain: OutGain));
                t.Add(Norm(p + "ple_norm_key.weight", hcDim));
                t.Add(Norm(p + "ple_norm_query.weight", hcDim));
                t.Add(Norm(p + "ple_norm_conv.weight", hcDim));
            }

            // The shipped file's mix: IQ3_S gate/up with IQ4_NL down, one layer of IQ4_XS gate/up,
            // and a few layers of Q8_0 down.
            GgmlType gateUp = il == 2 ? GgmlType.IQ4_XS : GgmlType.IQ3_S;
            GgmlType down = il is 2 or 6 ? GgmlType.Q8_0 : GgmlType.IQ4_NL;
            if (q2kxlExperts)
            {
                gateUp = il == 2 ? GgmlType.IQ3_XXS : GgmlType.IQ2_XS;
                down = GgmlType.IQ4_NL;
            }
            if (q4kxlExperts)
            {
                gateUp = GgmlType.Q4_K;
                down = GgmlType.Q5_1;
            }
            if (expertGateUpType is { } gateUpOverride)
                gateUp = gateUpOverride;
            if (expertDownType is { } downOverride)
                down = downOverride;
            t.Add(Proj(p + "ffn_gate_inp.weight", hiddenSize, expertCount, routerType));
            t.Add(Experts3(p + "ffn_gate_exps.weight", hiddenSize, expertFf, gateUp));
            t.Add(Experts3(p + "ffn_up_exps.weight", hiddenSize, expertFf, gateUp));
            t.Add(Experts3(p + "ffn_down_exps.weight", expertFf, hiddenSize, down, gain: expertDownGain));
            t.Add(Gen(p + "ffn_gate_inp_shexp.weight", 1.5f / System.MathF.Sqrt(hiddenSize), hiddenSize));
            t.Add(Proj(p + "ffn_gate_shexp.weight", hiddenSize, SharedFf));
            t.Add(Proj(p + "ffn_up_shexp.weight", hiddenSize, SharedFf));
            t.Add(Proj(p + "ffn_down_shexp.weight", SharedFf, hiddenSize, gain: OutGain));
        }

        var kv = Metadata(Layers, indexerTopK, contextLength, qsa, expertCount, expertUsedCount, hiddenSize, expertFf,
            pleHeadDim, pleConvKernel, ssmConvKernel);
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    /// <summary>The checkpoint's metadata, <paramref name="blockCount"/> blocks (a NextN file counts its block).</summary>
    private static List<Kv> Metadata(int blockCount, int indexerTopK, int contextLength, bool qsa, int expertCount,
        int expertUsedCount, int hiddenSize, int expertFf, int pleHeadDim, int pleConvKernel, int ssmConvKernel)
    {
        const int valueDim = StateSize * VHeads;
        const string a = "qwen4exp.";
        var ratios = new int[blockCount];
        var pleOffsets = new ulong[PleVocab.Length];
        for (int il = 0; il < Layers; il++) ratios[il] = IsRecurrent(il) ? 0 : CompressRatio;
        for (int h = 1; h < PleVocab.Length; h++) pleOffsets[h] = pleOffsets[h - 1] + PleVocab[h - 1];
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = "qwen4exp" },
            new Str { Key = "general.name", V = "Synthetic qwen4exp engine fixture" },
            new U32 { Key = a + "block_count", V = (uint)blockCount },
            new U32 { Key = a + "context_length", V = (uint)contextLength },
            new U32 { Key = a + "embedding_length", V = (uint)hiddenSize },
            new U32 { Key = a + "attention.head_count", V = Heads },
            new U32 { Key = a + "attention.head_count_kv", V = KvHeads },
            new I32Arr { Key = a + "rope.dimension_sections", V = new[] { 11, 11, 10, 0 } },
            new F32 { Key = a + "rope.freq_base", V = 1e7f },
            new F32 { Key = a + "attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = a + "expert_count", V = (uint)expertCount },
            new U32 { Key = a + "expert_used_count", V = (uint)expertUsedCount },
            new U32 { Key = a + "attention.key_length", V = HeadDim },
            new U32 { Key = a + "attention.value_length", V = HeadDim },
            new U32 { Key = a + "expert_feed_forward_length", V = (uint)expertFf },
            new U32 { Key = a + "expert_shared_feed_forward_length", V = SharedFf },
            new U32 { Key = a + "ssm.conv_kernel", V = (uint)ssmConvKernel },
            new U32 { Key = a + "ssm.state_size", V = StateSize },
            new U32 { Key = a + "ssm.group_count", V = KHeads },
            new U32 { Key = a + "ssm.time_step_rank", V = VHeads },
            new U32 { Key = a + "ssm.inner_size", V = valueDim },
            new U32 { Key = a + "full_attention_interval", V = 4 },
            new U32 { Key = a + "rope.dimension_count", V = RopeDims },
            new U32 { Key = a + "hyper_connection.count", V = Hc },
            new U32 { Key = a + "hyper_connection.low_rank", V = HcLowRank },
            new U32 { Key = a + "attention.indexer.head_count", V = IndexerHeads },
            new U32 { Key = a + "attention.indexer.key_length", V = IndexerDim },
            new U32 { Key = a + "attention.indexer.top_k", V = (uint)indexerTopK },
            new I32Arr { Key = a + "attention.compress_ratios", V = qsa ? ratios : new int[blockCount] },
            new I32Arr { Key = a + "ple.layers", V = new[] { PleLayer } },
            new U32 { Key = a + "ple.ngram_size", V = PleNgram },
            new U32 { Key = a + "ple.heads_per_ngram", V = PleHeadsPerNgram },
            new U32 { Key = a + "ple.conv_kernel", V = (uint)pleConvKernel },
            new U32 { Key = a + "ple.eos_token_id", V = EosToken },
            new U32 { Key = a + "embedding_length_per_layer_input", V = (uint)pleHeadDim },
            // The shipped multipliers: the hash is exercised at its real 64-bit width.
            new U64Arr { Key = a + "ple.layer_multipliers", V = new ulong[] { 23703573157769, 20109073645365, 8052911324071 } },
            new U64Arr { Key = a + "ple.head_offsets", V = pleOffsets },
            new U64Arr { Key = a + "ple.head_vocab_sizes", V = PleVocab },
        };
        AddByteTokenizer(kv, Vocab, BosToken, EosToken);
        return kv;
    }

    /// <summary>
    /// A shared-target MTP (NextN) file for the checkpoint <see cref="Write"/> writes with its defaults: block
    /// <see cref="Layers"/> with the published head's 32 tensors (Q8_0 matrices, F32 norms, the inactive BF16
    /// indexer), and the checkpoint's metadata and tokenizer, as the shipped head carries them.
    /// </summary>
    public static string WriteMtp(string path, int indexerTopK = 16, int contextLength = 4096)
    {
        const int hiddenSize = Hidden, hcDim = Hc * Hidden;
        string p = $"blk.{Layers}.";
        var t = new List<Tensor>();
        Tensor Proj(string name, int inDim, int outDim, GgmlType type = GgmlType.Q8_0, float gain = 1f)
        {
            var w = Gen(name, gain * 1.5f / System.MathF.Sqrt(inDim), inDim, outDim);
            w.Type = type;
            return w;
        }
        Tensor Norm(string name, int n) => GenAround(name, 1f, 0.1f, n);
        const float OutGain = 0.25f;
        t.Add(Norm(p + "nextn.enorm.weight", hiddenSize));
        t.Add(Norm(p + "nextn.hnorm.weight", hcDim));
        t.Add(Proj(p + "nextn.eh_proj.weight", 2 * hiddenSize, hiddenSize));
        t.Add(Norm(p + "nextn.hc_head_norm.weight", hcDim));
        t.Add(Proj(p + "nextn.hc_head_down.weight", hcDim, HcLowRank));
        t.Add(Proj(p + "nextn.hc_head_up.weight", HcLowRank, hcDim));
        foreach (string part in new[] { "attn", "ffn" })
        {
            t.Add(Norm(p + $"hc_{part}_norm.weight", hcDim));
            t.Add(Proj(p + $"hc_{part}_down.weight", hcDim, HcLowRank));
            t.Add(Proj(p + $"hc_{part}_up.weight", HcLowRank, hcDim));
            t.Add(Proj(p + $"hc_{part}_inject.weight", hcDim, Hc));
        }
        t.Add(Proj(p + "attn_q.weight", hiddenSize, Heads * 2 * HeadDim));
        t.Add(Proj(p + "attn_k.weight", hiddenSize, KvHeads * HeadDim));
        t.Add(Proj(p + "attn_v.weight", hiddenSize, KvHeads * HeadDim));
        t.Add(Norm(p + "attn_q_norm.weight", HeadDim));
        t.Add(Norm(p + "attn_k_norm.weight", HeadDim));
        t.Add(Proj(p + "attn_output.weight", Heads * HeadDim, hiddenSize, gain: OutGain));
        t.Add(Proj(p + "indexer.k_proj.weight", hiddenSize, IndexerDim, GgmlType.BF16));
        t.Add(Proj(p + "indexer.q_proj.weight", hiddenSize, IndexerHeads * IndexerDim, GgmlType.BF16));
        t.Add(Norm(p + "indexer.k_norm.weight", IndexerDim));
        t.Add(Norm(p + "indexer.q_norm.weight", IndexerDim));
        t.Add(Proj(p + "ffn_gate_inp.weight", hiddenSize, Experts, GgmlType.F32));
        var gate = Gen(p + "ffn_gate_exps.weight", 1.5f / System.MathF.Sqrt(hiddenSize), hiddenSize, ExpertFf, Experts);
        var up = Gen(p + "ffn_up_exps.weight", 1.5f / System.MathF.Sqrt(hiddenSize), hiddenSize, ExpertFf, Experts);
        var down = Gen(p + "ffn_down_exps.weight", 0.25f * 1.5f / System.MathF.Sqrt(ExpertFf), ExpertFf, hiddenSize, Experts);
        gate.Type = up.Type = down.Type = GgmlType.Q8_0;
        t.Add(gate); t.Add(up); t.Add(down);
        t.Add(Gen(p + "ffn_gate_inp_shexp.weight", 1.5f / System.MathF.Sqrt(hiddenSize), hiddenSize));
        t.Add(Proj(p + "ffn_gate_shexp.weight", hiddenSize, SharedFf));
        t.Add(Proj(p + "ffn_up_shexp.weight", hiddenSize, SharedFf));
        t.Add(Proj(p + "ffn_down_shexp.weight", SharedFf, hiddenSize, gain: OutGain));

        int pleHeadDim = hiddenSize / ((PleNgram - 1) * PleHeadsPerNgram);
        var kv = Metadata(Layers + 1, indexerTopK, contextLength, qsa: true, Experts, ExpertsUsed, hiddenSize, ExpertFf,
            pleHeadDim, ConvKernel, ConvKernel);
        kv.Add(new Bool { Key = "qwen4exp.nextn_shared_target_tensors", V = true });
        kv.Add(new U32 { Key = "qwen4exp.nextn_predict_layers", V = 1 });
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    /// <summary>A one-layer DFlash drafter for the checkpoint <see cref="Write"/> writes: it reads the stream entering
    /// layers [2, 5, 7], as Flash Next's DFlash2 reads [6, 16, 26, 36, 46], and borrows the trunk's embedding and head.
    /// <paramref name="hidden"/>: another trunk's width (a Qwen 3.5 synthetic checkpoint's).</summary>
    public static string WriteDFlash(string path, int blockSize = 4, int hidden = Hidden)
    {
        const int heads = 4, kvHeads = 2, headDim = 64, ffn = 512;
        int Hidden = hidden;
        int[] taps = { 2, 5, 7 };
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = "dflash" },
            new U32 { Key = "dflash.block_count", V = 1 },
            new U32 { Key = "dflash.embedding_length", V = (uint)Hidden },
            new U32 { Key = "dflash.feed_forward_length", V = ffn },
            new U32 { Key = "dflash.attention.head_count", V = heads },
            new U32 { Key = "dflash.attention.head_count_kv", V = kvHeads },
            new U32 { Key = "dflash.attention.key_length", V = headDim },
            new U32 { Key = "dflash.attention.value_length", V = headDim },
            new F32 { Key = "dflash.rope.freq_base", V = 1e7f },
            new F32 { Key = "dflash.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new U32 { Key = "dflash.block_size", V = (uint)blockSize },
            new I32Arr { Key = "dflash.target_layers", V = taps },
            new U32 { Key = "dflash.attention.sliding_window", V = 4096 },
            new U32 { Key = "tokenizer.ggml.mask_token_id", V = Vocab - 3 },
        };
        float S(int inDim) => 1.5f / System.MathF.Sqrt(inDim);
        var t = new List<Tensor>
        {
            Gen("fc.weight", S(taps.Length * Hidden), taps.Length * Hidden, Hidden),
            GenAround("enc.output_norm.weight", 1f, 0.1f, Hidden),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
            GenAround("blk.0.attn_norm.weight", 1f, 0.1f, Hidden),
            Gen("blk.0.attn_q.weight", S(Hidden), Hidden, heads * headDim),
            Gen("blk.0.attn_k.weight", S(Hidden), Hidden, kvHeads * headDim),
            Gen("blk.0.attn_v.weight", S(Hidden), Hidden, kvHeads * headDim),
            Gen("blk.0.attn_output.weight", 0.25f * S(heads * headDim), heads * headDim, Hidden),
            GenAround("blk.0.attn_q_norm.weight", 1f, 0.1f, headDim),
            GenAround("blk.0.attn_k_norm.weight", 1f, 0.1f, headDim),
            GenAround("blk.0.ffn_norm.weight", 1f, 0.1f, Hidden),
            Gen("blk.0.ffn_gate.weight", S(Hidden), Hidden, ffn),
            Gen("blk.0.ffn_up.weight", S(Hidden), Hidden, ffn),
            Gen("blk.0.ffn_down.weight", 0.25f * S(ffn), ffn, Hidden),
        };
        SyntheticGguf.Write(path, kv, t);
        return path;
    }
}
