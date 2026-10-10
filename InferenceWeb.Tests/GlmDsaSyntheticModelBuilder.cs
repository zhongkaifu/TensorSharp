// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Builds a tiny, fully-deterministic glm-dsa GGUF in memory.
//
// GLM-5.2 is 226 GiB, so it can only ever be an opt-in test. This builder makes
// the ARCHITECTURE testable without it: same block shape (MLA with wk_b/wv_b
// absorption, a DSA lightning indexer, sigmoid-gated MoE with a shared expert,
// a leading dense layer and a trailing NextN block), 1.9 MB of F32 weights, and
// a deterministic generator so llama.cpp and TensorSharp see bit-identical
// inputs. The indexer's top_k is deliberately tiny (8) so a 24-token prompt
// already exercises the SPARSE attention path, which the real model only
// reaches past 2048 tokens.
//
// The goldens in GlmDsaTinyModelTests were captured from llama.cpp built at the
// commit recorded there; regenerate them with .parity/glm_parity.cpp if the
// reference implementation changes.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace InferenceWeb.Tests;

internal static class GlmDsaSyntheticModelBuilder
{
    public const int NumBlocks = 5;          // 4 trunk + 1 NextN
    public const int NumNextn = 1;
    public const int DenseLead = 1;
    public const int Hidden = 64;
    public const int FfnLength = 128;
    public const int NumHeads = 4;
    public const int RopeDim = 16;
    public const int KvLoraRank = 32;
    public const int QLoraRank = 32;
    public const int HeadDimK = 48;          // key_length_mla
    public const int HeadDimV = 48;          // value_length_mla
    public const int NumExperts = 8;
    public const int ExpertsUsed = 2;
    public const int ExpertFfn = 32;
    public const int IndexerHeads = 8;
    public const int IndexerDim = 64;        // must be a power of two >= 64 (Hadamard)
    public const int IndexerTopK = 8;
    public const int VocabSize = 128;
    public const int ContextLength = 4096;

    private const int Nope = HeadDimK - RopeDim;

    /// <summary>Write the model to <paramref name="path"/>; returns the path.</summary>
    /// <param name="quantize">
    /// Emit the 2D/3D weight matrices as Q8_0 instead of F32, so the quantized
    /// matmul routers (`ManagedQuantizedOps` / `GgmlBasicOps.AddmmQuant` /
    /// `mul_mat_id` over quantized experts) are covered too — the real
    /// checkpoints are never F32. Every weight's leading dimension is a multiple
    /// of 32 for exactly this reason. 1D tensors (norms, biases) stay F32, as
    /// llama-quantize leaves them.
    /// </param>
    public static string Write(string path, int[] indexerTypes = null, bool quantize = false)
    {
        var tensors = BuildTensors();
        var kv = BuildMetadata(indexerTypes);
        if (quantize)
            foreach (var t in tensors)
                if (t.Dims.Length >= 2) t.Type = GgmlType.Q8_0;

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    /// <summary>
    /// Write a small glm5next KDA model for native tensor-parallel loader
    /// tests and the speculative rollback tests. The dimensions are deliberately
    /// tiny, but preserve the production partitioning constraint: a 16-wide KDA
    /// head in a Q8_0 output projection must travel in two-head groups because
    /// Q8_0 has 32-element blocks. By default every block is a KDA (recurrent)
    /// layer with a dense FFN. Optional mixed MLA attention and routed experts
    /// exercise verification, indexing, and quantized projection parity.
    /// </summary>
    public static string WriteGlm5NextTpFixture(
        string path, int numHeads, bool quantizeAttentionOutput, int numLayers = 1,
        bool mixedAttention = false, bool routedExperts = false, bool quantizeExperts = false,
        int contextLength = 256, int indexerTopK = 4, string architecture = "glm5next")
    {
        const int hidden = 64;
        const int ffn = 64;
        const int headDim = 16;
        const int lowRank = 32;
        const int conv = 4;
        const int hc = 4;
        const int vocab = 128;
        int dInner = numHeads * headDim;

        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (contextLength <= 0 || indexerTopK <= 0 || indexerTopK % 4 != 0)
            throw new ArgumentException("Context must be positive and pooled indexer top-k must be a positive multiple of four.");
        if (mixedAttention && numLayers < 2)
            throw new ArgumentException("Mixed attention requires at least one KDA and one MLA layer.");
        if (quantizeExperts && !routedExperts)
            throw new ArgumentException("Quantized experts require routedExperts.");
        if (quantizeAttentionOutput && dInner % Q8Block != 0)
            throw new ArgumentException("The Q8_0 fixture's KDA width must contain whole quantization blocks.",
                nameof(numHeads));

        var tensors = new List<TensorSpec>
        {
            Gen("token_embd.weight", 0.02f, hidden, vocab),
            Gen("output_norm.weight", 0.2f, hidden),
            Gen("output.weight", 0.02f, hidden, vocab),
        };
        for (int l = 0; l < numLayers; l++)
        {
            string p = $"blk.{l}.";
            tensors.AddRange(new[]
            {
                Gen(p + "attn_norm.weight", 0.2f, hidden),
                Gen(p + "ffn_norm.weight", 0.2f, hidden),

                Gen(p + "hc_attn_fn.weight", 0.02f, hc * hidden, (2 + hc) * hc),
                Gen(p + "hc_attn_scale.weight", 0.02f, 3),
                Gen(p + "hc_attn_base.weight", 0.02f, (2 + hc) * hc),
                Gen(p + "hc_ffn_fn.weight", 0.02f, hc * hidden, (2 + hc) * hc),
                Gen(p + "hc_ffn_scale.weight", 0.02f, 3),
                Gen(p + "hc_ffn_base.weight", 0.02f, (2 + hc) * hc),

                Gen(p + "attn_q.weight", 0.02f, hidden, dInner),
                Gen(p + "attn_k.weight", 0.02f, hidden, dInner),
                Gen(p + "attn_v.weight", 0.02f, hidden, dInner),
                Gen(p + "ssm_conv1d_q.weight", 0.02f, conv, 1, dInner),
                Gen(p + "ssm_conv1d_k.weight", 0.02f, conv, 1, dInner),
                Gen(p + "ssm_conv1d_v.weight", 0.02f, conv, 1, dInner),
                Gen(p + "ssm_f_a.weight", 0.02f, hidden, lowRank),
                Gen(p + "ssm_f_b.weight", 0.02f, lowRank, dInner),
                Gen(p + "ssm_dt.bias", 0.02f, dInner),
                Gen(p + "ssm_a", 0.02f, numHeads),
                Gen(p + "ssm_beta.weight", 0.02f, hidden, numHeads),
                Gen(p + "ssm_g_a.weight", 0.02f, hidden, lowRank),
                Gen(p + "ssm_g_b.weight", 0.02f, lowRank, dInner),
                Gen(p + "ssm_norm.weight", 0.2f, headDim),

                Gen(p + "ffn_gate.weight", 0.02f, hidden, ffn),
                Gen(p + "ffn_up.weight", 0.02f, hidden, ffn),
                Gen(p + "ffn_down.weight", 0.02f, ffn, hidden),
            });
            var attentionOutput = Gen(p + "attn_output.weight", 0.02f, dInner, hidden);
            if (quantizeAttentionOutput)
                attentionOutput.Type = GgmlType.Q8_0;
            tensors.Add(attentionOutput);
        }

        // Optional verification fixture: preserve the existing all-KDA/dense
        // defaults byte-for-byte while exercising the production mixed layout.
        var kvHeads = new uint[numLayers];
        const int expertFfn = 64; // TP2 leaves a whole Q8_0 block per rank.
        for (int l = 0; l < numLayers; l++)
        {
            string p = $"blk.{l}.";
            if (mixedAttention && l % 2 == 1)
            {
                kvHeads[l] = 1;
                tensors.RemoveAll(t => t.Name.StartsWith(p + "ssm_", StringComparison.Ordinal)
                    || t.Name == p + "attn_q.weight" || t.Name == p + "attn_k.weight"
                    || t.Name == p + "attn_v.weight" || t.Name == p + "attn_output.weight");
                tensors.AddRange(new[]
                {
                    Gen(p + "attn_q_a.weight", 0.02f, hidden, lowRank),
                    Gen(p + "attn_q_a_norm.weight", 0.2f, lowRank),
                    Gen(p + "attn_q_b.weight", 0.02f, lowRank, numHeads * lowRank),
                    Gen(p + "attn_kv_a_mqa.weight", 0.02f, hidden, lowRank),
                    Gen(p + "attn_kv_a_norm.weight", 0.2f, lowRank),
                    Gen(p + "attn_k_b.weight", 0.02f, lowRank, lowRank, numHeads),
                    Gen(p + "attn_v_b.weight", 0.02f, lowRank, lowRank, numHeads),
                    Gen(p + "indexer.attn_q_b.weight", 0.02f, lowRank, 64),
                    Gen(p + "indexer.attn_k.weight", 0.02f, hidden, 64),
                    Gen(p + "indexer.k_norm.weight", 0.2f, 64),
                    Gen(p + "indexer.k_norm.bias", 0.02f, 64),
                    Gen(p + "indexer.proj.weight", 0.02f, hidden, 1),
                    Gen(p + "indexer_compressor_gate.weight", 0.02f, hidden, 64),
                    Gen(p + "indexer_compressor_ape.weight", 0.02f, 64, 4),
                });
                var output = Gen(p + "attn_output.weight", 0.02f, numHeads * lowRank, hidden);
                if (quantizeAttentionOutput) output.Type = GgmlType.Q8_0;
                tensors.Add(output);
            }
            if (routedExperts)
            {
                tensors.RemoveAll(t => t.Name == p + "ffn_gate.weight"
                    || t.Name == p + "ffn_up.weight" || t.Name == p + "ffn_down.weight");
                tensors.AddRange(new[]
                {
                    Gen(p + "ffn_gate_inp.weight", 0.2f, hidden, 4),
                    Gen(p + "exp_probs_b.bias", 0.02f, 4),
                    Gen(p + "ffn_gate_exps.weight", 0.02f, hidden, expertFfn, 4),
                    Gen(p + "ffn_up_exps.weight", 0.02f, hidden, expertFfn, 4),
                    Gen(p + "ffn_down_exps.weight", 0.02f, expertFfn, hidden, 4),
                    Gen(p + "ffn_gate_shexp.weight", 0.02f, hidden, expertFfn),
                    Gen(p + "ffn_up_shexp.weight", 0.02f, hidden, expertFfn),
                    Gen(p + "ffn_down_shexp.weight", 0.02f, expertFfn, hidden),
                });
            }
        }
        if (quantizeExperts)
            foreach (var tensor in tensors)
                if (tensor.Name.EndsWith("_exps.weight", StringComparison.Ordinal)) tensor.Type = GgmlType.Q8_0;

        if (architecture is not ("glm5next" or "glm5-next"))
            throw new ArgumentException("Expected a GLM-5-Next GGUF architecture", nameof(architecture));
        string a = architecture;
        var kv = new List<KvEntry>
        {
            new KvStr  { Key = "general.architecture", V = a },
            new KvStr  { Key = "general.name", V = "tiny-glm5next-tp" },
            new KvU32  { Key = $"{a}.block_count", V = (uint)numLayers },
            new KvU32  { Key = $"{a}.context_length", V = (uint)contextLength },
            new KvU32  { Key = $"{a}.embedding_length", V = hidden },
            new KvU32  { Key = $"{a}.feed_forward_length", V = ffn },
            new KvU32  { Key = $"{a}.attention.head_count", V = (uint)numHeads },
            new KvU32Arr { Key = $"{a}.attention.head_count_kv", V = kvHeads },
            new KvF32  { Key = $"{a}.rope.freq_base", V = 10000.0f },
            new KvF32  { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new KvF32  { Key = $"{a}.attention.layer_norm_epsilon", V = 1e-6f },
            new KvU32  { Key = $"{a}.attention.key_length", V = lowRank },
            new KvU32  { Key = $"{a}.attention.value_length", V = lowRank },
            new KvU32  { Key = $"{a}.attention.q_lora_rank", V = lowRank },
            new KvU32  { Key = $"{a}.attention.kv_lora_rank", V = lowRank },
            new KvU32  { Key = $"{a}.attention.key_length_mla", V = lowRank },
            new KvU32  { Key = $"{a}.attention.value_length_mla", V = lowRank },
            new KvU32  { Key = $"{a}.rope.dimension_count", V = 0 },

            // These model-level fields are also required by the default
            // all-KDA/dense fixture.
            new KvU32  { Key = $"{a}.leading_dense_block_count", V = routedExperts ? 0u : (uint)numLayers },
            new KvU32  { Key = $"{a}.expert_count", V = 4 },
            new KvU32  { Key = $"{a}.expert_used_count", V = 2 },
            new KvU32  { Key = $"{a}.expert_feed_forward_length", V = (uint)(routedExperts ? expertFfn : 32) },
            new KvU32  { Key = $"{a}.expert_group_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_group_used_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_gating_func", V = 2 },
            new KvU32  { Key = $"{a}.attention.indexer.head_count", V = 1 },
            new KvU32  { Key = $"{a}.attention.indexer.key_length", V = 64 },
            new KvU32  { Key = $"{a}.attention.indexer.top_k", V = (uint)indexerTopK },
            new KvU32  { Key = $"{a}.attention.indexer.kpool", V = 4 },

            new KvU32  { Key = $"{a}.kda.head_dim", V = headDim },
            new KvU32  { Key = $"{a}.ssm.conv_kernel", V = conv },
            new KvF32  { Key = $"{a}.kda.gate_lower_bound", V = -5.0f },
            new KvU32  { Key = $"{a}.hyper_connection.count", V = hc },
            new KvU32  { Key = $"{a}.hyper_connection.sinkhorn_iterations", V = 4 },
            new KvF32  { Key = $"{a}.hyper_connection.epsilon", V = 1e-6f },
            new KvU32  { Key = $"{a}.vocab_size", V = vocab },
        };

        var tokens = new List<string>(vocab);
        foreach (int b in ByteToUnicode(vocab)) tokens.Add(char.ConvertFromUtf32(b));
        var types = new int[vocab];
        Array.Fill(types, 1);
        kv.Add(new KvStr { Key = "tokenizer.ggml.model", V = "gpt2" });
        kv.Add(new KvStr { Key = "tokenizer.ggml.pre", V = "gpt-2" });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.tokens", V = tokens });
        kv.Add(new KvI32Arr { Key = "tokenizer.ggml.token_type", V = types });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.merges", V = Array.Empty<string>() });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.bos_token_id", V = 'A' });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.eos_token_id", V = 'Z' });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_bos_token", V = false });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_eos_token", V = false });
        kv.Add(new KvU32 { Key = "general.file_type", V = quantizeAttentionOutput || quantizeExperts ? 7u : 0u });
        kv.Add(new KvU32 { Key = "general.quantization_version", V = 2 });

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    /// <summary>
    /// A small glm5next model with the production model's widths where the direct-CUDA engine's
    /// kernels are specialized: KDA heads 128 wide, a 512-wide MLA latent, a 128-wide indexer key,
    /// 4 hyper-connection streams over a hidden width whose flattening is a multiple of 256, and
    /// expert widths that are multiples of 128 (the tensor-core expert kernel's k-step). Four
    /// trunk layers (KDA with a dense FFN, then MLA, KDA, MLA over 8 routed experts), every
    /// matrix Q8_0 as in the published checkpoint's dense projections, a SwiGLU clamp, and an
    /// indexer top-k past any prompt the tests use (the dense attention both engines run there).
    /// The indexer has 16 heads (the checkpoint has 32): a pool scores exactly 0 when every head's
    /// dot product is negative, and with two heads a quarter of the pools would, leaving the
    /// selection mostly to tie order.
    /// </summary>
    /// <param name="indexerTopK">Cells the indexer keeps per query once it selects; the default is
    /// past any prompt the tests use (dense attention), a small value makes the pooled selection
    /// run from the first ubatch.</param>
    public static string WriteGlm5NextCudaFixture(string path, int contextLength = 512, int indexerTopK = 2048)
    {
        const string a = "glm5next";
        const int hidden = 256, heads = 2, headDim = 128, lowRank = 64, conv = 4, hc = 4, vocab = 256;
        const int qLora = 64, kvLora = 512, headK = 128, headV = 128;
        const int idxHeads = 16, idxDim = 128, experts = 8, used = 2, expertFfn = 256, ffn = 256, layers = 4;
        int dInner = heads * headDim;
        var tensors = new List<TensorSpec>
        {
            Gen("token_embd.weight", 0.5f, hidden, vocab),
            Gen("output_norm.weight", 0.5f, hidden),
            Gen("output.weight", 0.08f, hidden, vocab),
        };
        var kvHeads = new uint[layers];
        for (int l = 0; l < layers; l++)
        {
            string p = $"blk.{l}.";
            tensors.AddRange(new[]
            {
                Gen(p + "attn_norm.weight", 0.5f, hidden),
                Gen(p + "ffn_norm.weight", 0.5f, hidden),
                Gen(p + "hc_attn_fn.weight", 0.05f, hc * hidden, (2 + hc) * hc),
                Gen(p + "hc_attn_scale.weight", 0.2f, 3),
                Gen(p + "hc_attn_base.weight", 0.2f, (2 + hc) * hc),
                Gen(p + "hc_ffn_fn.weight", 0.05f, hc * hidden, (2 + hc) * hc),
                Gen(p + "hc_ffn_scale.weight", 0.2f, 3),
                Gen(p + "hc_ffn_base.weight", 0.2f, (2 + hc) * hc),
            });
            if (l % 2 == 0)
            {
                tensors.AddRange(new[]
                {
                    Gen(p + "attn_q.weight", 0.08f, hidden, dInner),
                    Gen(p + "attn_k.weight", 0.08f, hidden, dInner),
                    Gen(p + "attn_v.weight", 0.08f, hidden, dInner),
                    Gen(p + "ssm_conv1d_q.weight", 0.3f, conv, 1, dInner),
                    Gen(p + "ssm_conv1d_k.weight", 0.3f, conv, 1, dInner),
                    Gen(p + "ssm_conv1d_v.weight", 0.3f, conv, 1, dInner),
                    Gen(p + "ssm_f_a.weight", 0.08f, hidden, lowRank),
                    Gen(p + "ssm_f_b.weight", 0.08f, lowRank, dInner),
                    Gen(p + "ssm_dt.bias", 0.5f, dInner),
                    Gen(p + "ssm_a", 0.5f, heads),
                    Gen(p + "ssm_beta.weight", 0.08f, hidden, heads),
                    Gen(p + "ssm_g_a.weight", 0.08f, hidden, lowRank),
                    Gen(p + "ssm_g_b.weight", 0.08f, lowRank, dInner),
                    Gen(p + "ssm_norm.weight", 0.5f, headDim),
                    Gen(p + "attn_output.weight", 0.08f, dInner, hidden),
                });
            }
            else
            {
                kvHeads[l] = 1;
                tensors.AddRange(new[]
                {
                    Gen(p + "attn_q_a.weight", 0.08f, hidden, qLora),
                    Gen(p + "attn_q_a_norm.weight", 0.5f, qLora),
                    Gen(p + "attn_q_b.weight", 0.08f, qLora, heads * headK),
                    Gen(p + "attn_kv_a_mqa.weight", 0.08f, hidden, kvLora),
                    Gen(p + "attn_kv_a_norm.weight", 0.5f, kvLora),
                    Gen(p + "attn_k_b.weight", 0.08f, headK, kvLora, heads),
                    Gen(p + "attn_v_b.weight", 0.08f, kvLora, headV, heads),
                    Gen(p + "attn_output.weight", 0.08f, heads * headV, hidden),
                    Gen(p + "indexer.attn_q_b.weight", 0.08f, qLora, idxHeads * idxDim),
                    Gen(p + "indexer.attn_k.weight", 0.08f, hidden, idxDim),
                    Gen(p + "indexer.k_norm.weight", 0.5f, idxDim),
                    Gen(p + "indexer.k_norm.bias", 0.05f, idxDim),
                    Gen(p + "indexer.proj.weight", 0.2f, hidden, idxHeads),
                    Gen(p + "indexer_compressor_gate.weight", 0.08f, hidden, idxDim),
                    Gen(p + "indexer_compressor_ape.weight", 0.2f, idxDim, 4),
                });
            }
            if (l == 0)
            {
                tensors.AddRange(new[]
                {
                    Gen(p + "ffn_gate.weight", 0.08f, hidden, ffn),
                    Gen(p + "ffn_up.weight", 0.08f, hidden, ffn),
                    Gen(p + "ffn_down.weight", 0.08f, ffn, hidden),
                });
            }
            else
            {
                tensors.AddRange(new[]
                {
                    Gen(p + "ffn_gate_inp.weight", 0.3f, hidden, experts),
                    Gen(p + "exp_probs_b.bias", 0.1f, experts),
                    Gen(p + "ffn_gate_exps.weight", 0.08f, hidden, expertFfn, experts),
                    Gen(p + "ffn_up_exps.weight", 0.08f, hidden, expertFfn, experts),
                    Gen(p + "ffn_down_exps.weight", 0.08f, expertFfn, hidden, experts),
                    Gen(p + "ffn_gate_shexp.weight", 0.08f, hidden, expertFfn),
                    Gen(p + "ffn_up_shexp.weight", 0.08f, hidden, expertFfn),
                    Gen(p + "ffn_down_shexp.weight", 0.08f, expertFfn, hidden),
                });
            }
        }
        // As the published checkpoint stores them: matrices Q8_0 (the router stays F32), 1D F32.
        foreach (var t in tensors)
            if (t.Dims.Length >= 2 && !t.Name.EndsWith("ffn_gate_inp.weight", StringComparison.Ordinal)
                && !t.Name.Contains("ssm_conv1d", StringComparison.Ordinal)
                && !t.Name.EndsWith("indexer.proj.weight", StringComparison.Ordinal)
                && !t.Name.EndsWith("indexer_compressor_ape.weight", StringComparison.Ordinal))
                t.Type = GgmlType.Q8_0;

        var clamp = new float[layers];
        Array.Fill(clamp, 10.0f);
        var kv = new List<KvEntry>
        {
            new KvStr  { Key = "general.architecture", V = a },
            new KvStr  { Key = "general.name", V = "tiny-glm5next-cuda" },
            new KvU32  { Key = $"{a}.block_count", V = layers },
            new KvU32  { Key = $"{a}.context_length", V = (uint)contextLength },
            new KvU32  { Key = $"{a}.embedding_length", V = hidden },
            new KvU32  { Key = $"{a}.feed_forward_length", V = ffn },
            new KvU32  { Key = $"{a}.attention.head_count", V = heads },
            new KvU32Arr { Key = $"{a}.attention.head_count_kv", V = kvHeads },
            new KvF32  { Key = $"{a}.rope.freq_base", V = 10000.0f },
            new KvF32  { Key = $"{a}.attention.layer_norm_rms_epsilon", V = 1e-5f },
            new KvF32  { Key = $"{a}.attention.layer_norm_epsilon", V = 1e-6f },
            new KvU32  { Key = $"{a}.attention.key_length", V = kvLora },
            new KvU32  { Key = $"{a}.attention.value_length", V = kvLora },
            new KvU32  { Key = $"{a}.attention.q_lora_rank", V = qLora },
            new KvU32  { Key = $"{a}.attention.kv_lora_rank", V = kvLora },
            new KvU32  { Key = $"{a}.attention.key_length_mla", V = headK },
            new KvU32  { Key = $"{a}.attention.value_length_mla", V = headV },
            new KvU32  { Key = $"{a}.rope.dimension_count", V = 0 },
            new KvU32  { Key = $"{a}.leading_dense_block_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_count", V = experts },
            new KvU32  { Key = $"{a}.expert_used_count", V = used },
            new KvU32  { Key = $"{a}.expert_feed_forward_length", V = expertFfn },
            new KvU32  { Key = $"{a}.expert_shared_feed_forward_length", V = expertFfn },
            new KvU32  { Key = $"{a}.expert_shared_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_group_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_group_used_count", V = 1 },
            new KvU32  { Key = $"{a}.expert_gating_func", V = 2 },
            new KvF32  { Key = $"{a}.expert_weights_scale", V = 2.5f },
            new KvBool { Key = $"{a}.expert_weights_norm", V = true },
            new KvF32Arr { Key = $"{a}.swiglu_clamp_exp", V = clamp },
            new KvF32Arr { Key = $"{a}.swiglu_clamp_shexp", V = clamp },
            new KvU32  { Key = $"{a}.attention.indexer.head_count", V = idxHeads },
            new KvU32  { Key = $"{a}.attention.indexer.key_length", V = idxDim },
            new KvU32  { Key = $"{a}.attention.indexer.top_k", V = (uint)indexerTopK },
            new KvU32  { Key = $"{a}.attention.indexer.kpool", V = 4 },
            new KvU32  { Key = $"{a}.kda.head_dim", V = headDim },
            new KvU32  { Key = $"{a}.ssm.conv_kernel", V = conv },
            new KvF32  { Key = $"{a}.kda.gate_lower_bound", V = -5.0f },
            new KvU32  { Key = $"{a}.hyper_connection.count", V = hc },
            new KvU32  { Key = $"{a}.hyper_connection.sinkhorn_iterations", V = 20 },
            new KvF32  { Key = $"{a}.hyper_connection.epsilon", V = 1e-6f },
            new KvU32  { Key = $"{a}.vocab_size", V = vocab },
        };
        var tokens = new List<string>(vocab);
        foreach (int b in ByteToUnicode(vocab)) tokens.Add(char.ConvertFromUtf32(b));
        var types = new int[vocab];
        Array.Fill(types, 1);
        kv.Add(new KvStr { Key = "tokenizer.ggml.model", V = "gpt2" });
        kv.Add(new KvStr { Key = "tokenizer.ggml.pre", V = "gpt-2" });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.tokens", V = tokens });
        kv.Add(new KvI32Arr { Key = "tokenizer.ggml.token_type", V = types });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.merges", V = Array.Empty<string>() });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.bos_token_id", V = 'A' });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.eos_token_id", V = 'Z' });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_bos_token", V = false });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_eos_token", V = false });
        kv.Add(new KvU32 { Key = "general.file_type", V = 7u });
        kv.Add(new KvU32 { Key = "general.quantization_version", V = 2 });

        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    private enum GgmlType { F32 = 0, Q8_0 = 8 }

    private const int Q8Block = 32;
    private const int Q8BlockBytes = 2 + Q8Block;   // f16 scale + 32 int8

    /// <summary>
    /// ggml's `quantize_row_q8_0`, reproduced so the file is byte-identical to
    /// what `llama-quantize ... Q8_0` writes: per 32 values, scale = amax/127
    /// stored as F16, then round-to-nearest of x/scale.
    /// </summary>
    private static byte[] QuantizeQ8_0(float[] data, int rowLength)
    {
        int rows = data.Length / rowLength;
        int blocksPerRow = rowLength / Q8Block;
        var outBytes = new byte[(long)rows * blocksPerRow * Q8BlockBytes];
        int o = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * rowLength + b * Q8Block;
                float amax = 0;
                for (int j = 0; j < Q8Block; j++) amax = Math.Max(amax, Math.Abs(data[off + j]));
                float d = amax / 127.0f;
                ushort dHalf = FloatToHalfBits(d);
                float dRound = HalfBitsToFloat(dHalf);
                float id = dRound != 0 ? 1.0f / dRound : 0.0f;
                outBytes[o++] = (byte)(dHalf & 0xFF);
                outBytes[o++] = (byte)(dHalf >> 8);
                for (int j = 0; j < Q8Block; j++)
                    outBytes[o++] = unchecked((byte)(sbyte)Math.Round(data[off + j] * id, MidpointRounding.AwayFromZero));
            }
        }
        return outBytes;
    }

    private static ushort FloatToHalfBits(float f) => BitConverter.HalfToUInt16Bits((Half)f);
    private static float HalfBitsToFloat(ushort h) => (float)BitConverter.UInt16BitsToHalf(h);

    // --- deterministic weights ------------------------------------------------

    /// <summary>xorshift32 — identical output in any language, which is the point.</summary>
    private struct Rng
    {
        private uint _s;
        public Rng(uint seed) { _s = seed == 0 ? 0x9E3779B9u : seed; }
        public uint Next()
        {
            _s ^= _s << 13;
            _s ^= _s >> 17;
            _s ^= _s << 5;
            return _s;
        }
        /// <summary>Sum of four uniforms, centred: a rough bell shape in [-2, 2].</summary>
        public float Normalish(float scale)
        {
            double acc = 0;
            for (int i = 0; i < 4; i++) acc += Next() / 4294967296.0;
            return (float)((acc - 2.0) * scale);
        }
    }

    private static uint Fnv1a(string text)
    {
        uint h = 0x811c9dc5;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            h ^= b;
            h = unchecked(h * 0x01000193);
        }
        return h;
    }

    private sealed class TensorSpec
    {
        public string Name;
        public ulong[] Dims;     // GGUF order (ne0 fastest)
        public float[] Data;
        public GgmlType Type = GgmlType.F32;

        public byte[] Raw()
        {
            if (Type == GgmlType.Q8_0)
                return QuantizeQ8_0(Data, (int)Dims[0]);
            byte[] raw = new byte[Data.Length * sizeof(float)];
            Buffer.BlockCopy(Data, 0, raw, 0, raw.Length);
            return raw;
        }
    }

    private static TensorSpec Gen(string name, float scale, params int[] dims)
    {
        long n = 1;
        foreach (int d in dims) n *= d;
        var rng = new Rng(Fnv1a(name));
        var data = new float[n];
        for (long i = 0; i < n; i++) data[i] = rng.Normalish(scale);
        var ul = new ulong[dims.Length];
        for (int i = 0; i < dims.Length; i++) ul[i] = (ulong)dims[i];
        return new TensorSpec { Name = name, Dims = ul, Data = data };
    }

    private static List<TensorSpec> BuildTensors()
    {
        var t = new List<TensorSpec>
        {
            Gen("token_embd.weight", 0.08f, Hidden, VocabSize),
            Gen("output_norm.weight", 0.5f, Hidden),
            Gen("output.weight", 0.08f, Hidden, VocabSize),
        };

        for (int l = 0; l < NumBlocks; l++)
        {
            string p = $"blk.{l}.";
            t.Add(Gen(p + "attn_norm.weight", 0.5f, Hidden));
            t.Add(Gen(p + "attn_q_a.weight", 0.05f, Hidden, QLoraRank));
            t.Add(Gen(p + "attn_q_a_norm.weight", 0.5f, QLoraRank));
            t.Add(Gen(p + "attn_q_b.weight", 0.05f, QLoraRank, NumHeads * HeadDimK));
            t.Add(Gen(p + "attn_kv_a_mqa.weight", 0.05f, Hidden, KvLoraRank + RopeDim));
            t.Add(Gen(p + "attn_kv_a_norm.weight", 0.5f, KvLoraRank));
            t.Add(Gen(p + "attn_k_b.weight", 0.05f, Nope, KvLoraRank, NumHeads));
            t.Add(Gen(p + "attn_v_b.weight", 0.05f, KvLoraRank, HeadDimV, NumHeads));
            t.Add(Gen(p + "attn_output.weight", 0.05f, NumHeads * HeadDimV, Hidden));
            t.Add(Gen(p + "ffn_norm.weight", 0.5f, Hidden));
            t.Add(Gen(p + "indexer.attn_q_b.weight", 0.05f, QLoraRank, IndexerHeads * IndexerDim));
            t.Add(Gen(p + "indexer.attn_k.weight", 0.05f, Hidden, IndexerDim));
            t.Add(Gen(p + "indexer.k_norm.weight", 0.5f, IndexerDim));
            t.Add(Gen(p + "indexer.k_norm.bias", 0.02f, IndexerDim));
            t.Add(Gen(p + "indexer.proj.weight", 0.2f, Hidden, IndexerHeads));

            if (l < DenseLead)
            {
                t.Add(Gen(p + "ffn_gate.weight", 0.05f, Hidden, FfnLength));
                t.Add(Gen(p + "ffn_up.weight", 0.05f, Hidden, FfnLength));
                t.Add(Gen(p + "ffn_down.weight", 0.05f, FfnLength, Hidden));
            }
            else
            {
                t.Add(Gen(p + "ffn_gate_inp.weight", 0.2f, Hidden, NumExperts));
                t.Add(Gen(p + "exp_probs_b.bias", 0.1f, NumExperts));
                t.Add(Gen(p + "ffn_gate_exps.weight", 0.05f, Hidden, ExpertFfn, NumExperts));
                t.Add(Gen(p + "ffn_up_exps.weight", 0.05f, Hidden, ExpertFfn, NumExperts));
                t.Add(Gen(p + "ffn_down_exps.weight", 0.05f, ExpertFfn, Hidden, NumExperts));
                t.Add(Gen(p + "ffn_gate_shexp.weight", 0.05f, Hidden, ExpertFfn));
                t.Add(Gen(p + "ffn_up_shexp.weight", 0.05f, Hidden, ExpertFfn));
                t.Add(Gen(p + "ffn_down_shexp.weight", 0.05f, ExpertFfn, Hidden));
            }

            if (l >= NumBlocks - NumNextn)
            {
                t.Add(Gen(p + "nextn.eh_proj.weight", 0.05f, 2 * Hidden, Hidden));
                t.Add(Gen(p + "nextn.enorm.weight", 0.5f, Hidden));
                t.Add(Gen(p + "nextn.hnorm.weight", 0.5f, Hidden));
                t.Add(Gen(p + "nextn.shared_head_norm.weight", 0.5f, Hidden));
            }
        }
        return t;
    }

    // --- metadata -------------------------------------------------------------

    private abstract class KvEntry { public string Key; public abstract void Write(BinaryWriter w); }
    private sealed class KvU32 : KvEntry { public uint V; public override void Write(BinaryWriter w) { w.Write((uint)4); w.Write(V); } }
    private sealed class KvF32 : KvEntry { public float V; public override void Write(BinaryWriter w) { w.Write((uint)6); w.Write(V); } }
    private sealed class KvBool : KvEntry { public bool V; public override void Write(BinaryWriter w) { w.Write((uint)7); w.Write(V); } }
    private sealed class KvStr : KvEntry { public string V; public override void Write(BinaryWriter w) { w.Write((uint)8); WriteStr(w, V); } }
    private sealed class KvStrArr : KvEntry
    {
        public IReadOnlyList<string> V;
        public override void Write(BinaryWriter w)
        {
            w.Write((uint)9); w.Write((uint)8); w.Write((ulong)V.Count);
            foreach (string s in V) WriteStr(w, s);
        }
    }
    private sealed class KvI32Arr : KvEntry
    {
        public IReadOnlyList<int> V;
        public override void Write(BinaryWriter w)
        {
            w.Write((uint)9); w.Write((uint)5); w.Write((ulong)V.Count);
            foreach (int v in V) w.Write(v);
        }
    }
    private sealed class KvF32Arr : KvEntry
    {
        public IReadOnlyList<float> V;
        public override void Write(BinaryWriter w)
        {
            w.Write((uint)9); w.Write((uint)6); w.Write((ulong)V.Count);
            foreach (float v in V) w.Write(v);
        }
    }
    private sealed class KvU32Arr : KvEntry
    {
        public IReadOnlyList<uint> V;
        public override void Write(BinaryWriter w)
        {
            w.Write((uint)9); w.Write((uint)4); w.Write((ulong)V.Count);
            foreach (uint v in V) w.Write(v);
        }
    }

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    private static List<KvEntry> BuildMetadata(int[] indexerTypes)
    {
        const string A = "glm-dsa";
        var kv = new List<KvEntry>
        {
            new KvStr  { Key = "general.architecture", V = A },
            new KvStr  { Key = "general.name", V = "tiny-glm-dsa" },
            new KvU32  { Key = $"{A}.block_count", V = NumBlocks },
            new KvU32  { Key = $"{A}.context_length", V = ContextLength },
            new KvU32  { Key = $"{A}.embedding_length", V = Hidden },
            new KvU32  { Key = $"{A}.feed_forward_length", V = FfnLength },
            new KvU32  { Key = $"{A}.attention.head_count", V = NumHeads },
            new KvU32  { Key = $"{A}.attention.head_count_kv", V = 1 },
            new KvF32  { Key = $"{A}.rope.freq_base", V = 10000.0f },
            new KvF32  { Key = $"{A}.attention.layer_norm_rms_epsilon", V = 1e-6f },
            new KvU32  { Key = $"{A}.expert_count", V = NumExperts },
            new KvU32  { Key = $"{A}.expert_used_count", V = ExpertsUsed },
            new KvU32  { Key = $"{A}.expert_group_count", V = 1 },
            new KvU32  { Key = $"{A}.expert_group_used_count", V = 1 },
            new KvU32  { Key = $"{A}.expert_gating_func", V = 2 },
            new KvU32  { Key = $"{A}.attention.key_length", V = KvLoraRank + RopeDim },
            new KvU32  { Key = $"{A}.attention.value_length", V = KvLoraRank },
            new KvU32  { Key = $"{A}.leading_dense_block_count", V = DenseLead },
            new KvU32  { Key = $"{A}.vocab_size", V = VocabSize },
            new KvU32  { Key = $"{A}.attention.q_lora_rank", V = QLoraRank },
            new KvU32  { Key = $"{A}.attention.kv_lora_rank", V = KvLoraRank },
            new KvU32  { Key = $"{A}.attention.key_length_mla", V = HeadDimK },
            new KvU32  { Key = $"{A}.attention.value_length_mla", V = HeadDimV },
            new KvU32  { Key = $"{A}.expert_feed_forward_length", V = ExpertFfn },
            new KvU32  { Key = $"{A}.expert_shared_count", V = 1 },
            new KvF32  { Key = $"{A}.expert_weights_scale", V = 2.5f },
            new KvBool { Key = $"{A}.expert_weights_norm", V = true },
            new KvU32  { Key = $"{A}.rope.dimension_count", V = RopeDim },
            new KvU32  { Key = $"{A}.nextn_predict_layers", V = NumNextn },
            new KvU32  { Key = $"{A}.attention.indexer.head_count", V = IndexerHeads },
            new KvU32  { Key = $"{A}.attention.indexer.key_length", V = IndexerDim },
            new KvU32  { Key = $"{A}.attention.indexer.top_k", V = IndexerTopK },
        };

        if (indexerTypes != null && indexerTypes.Length > 0)
        {
            var arr = new uint[indexerTypes.Length];
            for (int i = 0; i < arr.Length; i++) arr[i] = (uint)indexerTypes[i];
            kv.Add(new KvU32Arr { Key = $"{A}.attention.indexer.types", V = arr });
        }

        // Byte-level BPE vocab whose token id IS the byte value, so an ASCII
        // prompt tokenizes to its own bytes in any correct implementation and the
        // test can feed explicit ids without depending on merge rules.
        var tokens = new List<string>(VocabSize);
        foreach (int b in ByteToUnicode(VocabSize)) tokens.Add(char.ConvertFromUtf32(b));
        var types = new int[VocabSize];
        for (int i = 0; i < VocabSize; i++) types[i] = 1;

        kv.Add(new KvStr { Key = "tokenizer.ggml.model", V = "gpt2" });
        kv.Add(new KvStr { Key = "tokenizer.ggml.pre", V = "gpt-2" });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.tokens", V = tokens });
        kv.Add(new KvI32Arr { Key = "tokenizer.ggml.token_type", V = types });
        kv.Add(new KvStrArr { Key = "tokenizer.ggml.merges", V = Array.Empty<string>() });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.bos_token_id", V = 'A' });
        kv.Add(new KvU32 { Key = "tokenizer.ggml.eos_token_id", V = 'Z' });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_bos_token", V = false });
        kv.Add(new KvBool { Key = "tokenizer.ggml.add_eos_token", V = false });
        kv.Add(new KvU32 { Key = "general.file_type", V = 0 });
        kv.Add(new KvU32 { Key = "general.quantization_version", V = 2 });
        return kv;
    }

    /// <summary>GPT-2's byte-to-unicode mapping, restricted to the first <paramref name="count"/> bytes.</summary>
    private static IEnumerable<int> ByteToUnicode(int count)
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b < 0xAD; b++) bs.Add(b);
        for (int b = 0xAE; b < 0x100; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bs.Contains(b)) continue;
            bs.Add(b);
            cs.Add(256 + n);
            n++;
        }
        var map = new int[256];
        for (int i = 0; i < bs.Count; i++) map[bs[i]] = cs[i];
        for (int i = 0; i < count; i++) yield return map[i];
    }

    // --- container ------------------------------------------------------------

    private const int Alignment = 32;

    private static void WriteGguf(Stream fs, List<KvEntry> kv, List<TensorSpec> tensors)
    {
        using var head = new MemoryStream();
        using (var w = new BinaryWriter(head, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);            // "GGUF"
            w.Write(3u);                      // version
            w.Write((ulong)tensors.Count);
            w.Write((ulong)kv.Count);

            foreach (var e in kv)
            {
                WriteStr(w, e.Key);
                e.Write(w);
            }

            ulong offset = 0;
            foreach (var t in tensors)
            {
                WriteStr(w, t.Name);
                w.Write((uint)t.Dims.Length);
                foreach (ulong d in t.Dims) w.Write(d);
                w.Write((uint)t.Type);
                w.Write(offset);
                ulong bytes = t.Type == GgmlType.Q8_0
                    ? (ulong)t.Data.Length / Q8Block * Q8BlockBytes
                    : (ulong)t.Data.Length * sizeof(float);
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        for (int i = 0; i < pad; i++) fs.WriteByte(0);

        var buffer = new byte[Alignment];
        foreach (var t in tensors)
        {
            byte[] raw = t.Raw();
            fs.Write(raw, 0, raw.Length);
            int tailPad = (Alignment - raw.Length % Alignment) % Alignment;
            if (tailPad > 0) fs.Write(buffer, 0, tailPad);
        }
    }
}
