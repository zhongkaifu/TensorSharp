// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>Format facts and checks of qwen4exp (Qwen 3.8 Flash Next) checkpoints shared by its loaders and
    /// executors: none of them hold model state.</summary>
    internal static class Qwen4ExpFormat
    {
        public const string ArchitectureId = "qwen4exp";

        private static readonly bool s_hcQ8 = Environment.GetEnvironmentVariable("TS_Q4E_HC_Q8") == "1";
        private static readonly System.Threading.AsyncLocal<bool?> s_hcQ8Flow = new();

        /// <summary>TS_Q4E_HC_Q8=1: BF16 hyper-connection matrices run as Q8_0 copies (a test's flow may choose for the
        /// models it constructs: <see cref="HcQ8ForCurrentFlowInTests"/>).</summary>
        internal static bool HcQ8 => s_hcQ8Flow.Value ?? s_hcQ8;

        /// <summary>Test seam: <see cref="HcQ8"/> for the models this flow constructs, until disposed.</summary>
        internal static IDisposable HcQ8ForCurrentFlowInTests(bool on)
        {
            bool? previous = s_hcQ8Flow.Value;
            s_hcQ8Flow.Value = on;
            return new FlowScope(() => s_hcQ8Flow.Value = previous);
        }

        private sealed class FlowScope : IDisposable
        {
            private readonly Action _restore;
            public FlowScope(Action restore) => _restore = restore;
            public void Dispose() => _restore();
        }

        /// <summary>True when <paramref name="path"/> names a DFlash/DFlash2 drafter (general.architecture = dflash) rather
        /// than a shared MTP head.</summary>
        internal static bool IsDFlashDraftFile(string path) => DFlashDraftFiles.Resolve(path) != null;

        // Validate metadata and tensor directories before mapping payloads or
        // borrowing target pointers. This does not read or claim weight values.
        // targetTensor finds the target's tensors when they are not in target's own
        // table: a split checkpoint opened shard by shard (the direct-CUDA engine's
        // GgufShardSet) keeps its metadata in the first shard and its tensors in the
        // others (Unsloth's six-shard Q8_0 has none in the first).
        internal static int ValidateMtpHead(GgufFile target, GgufFile draft, Func<string, GgufTensorInfo> targetTensor = null)
        {
            ArgumentNullException.ThrowIfNull(target);
            ArgumentNullException.ThrowIfNull(draft);
            targetTensor ??= name => target.Tensors.TryGetValue(name, out var t) ? t : null;
            const string a = "qwen4exp.";
            if (target.GetString("general.architecture") != ArchitectureId
                || draft.GetString("general.architecture") != ArchitectureId
                || !draft.GetBool(a + "nextn_shared_target_tensors")
                || draft.GetUint32(a + "nextn_predict_layers") != 1)
                throw new NotSupportedException("qwen4exp MTP requires one shared-target NextN block.");
            int trunk = checked((int)target.GetUint32(a + "block_count")
                - (int)target.GetUint32(a + "nextn_predict_layers"));
            int layer = checked((int)draft.GetUint32(a + "block_count") - 1);
            if (trunk <= 0 || layer != trunk)
                throw new InvalidOperationException("qwen4exp MTP block does not follow this target's trunk.");
            string[] integerKeys =
            {
                "embedding_length", "attention.head_count", "attention.head_count_kv",
                "attention.key_length", "attention.value_length", "expert_count", "expert_used_count",
                "expert_feed_forward_length", "expert_shared_feed_forward_length",
                "hyper_connection.count", "hyper_connection.low_rank", "rope.dimension_count",
                "attention.indexer.head_count", "attention.indexer.key_length", "attention.indexer.top_k",
            };
            foreach (string key in integerKeys)
                if (target.GetUint32(a + key) == 0 || target.GetUint32(a + key) != draft.GetUint32(a + key))
                    throw new InvalidOperationException($"qwen4exp MTP target mismatch: {key}.");
            foreach (string key in new[] { "rope.freq_base", "attention.layer_norm_rms_epsilon" })
                if (!target.Metadata.ContainsKey(a + key)
                    || target.GetFloat32(a + key) != draft.GetFloat32(a + key)
                    || !float.IsFinite(draft.GetFloat32(a + key)) || draft.GetFloat32(a + key) <= 0)
                    throw new InvalidOperationException($"qwen4exp MTP target mismatch: {key}.");
            var targetSections = target.GetInt32Array(a + "rope.dimension_sections");
            var draftSections = draft.GetInt32Array(a + "rope.dimension_sections");
            if (targetSections == null || draftSections == null || draftSections.Length != 4
                || draftSections.Any(x => x < 0)
                || !draftSections.Take(3).Any(x => x > 0)
                || draftSections.Sum(x => (long)x) > draft.GetUint32(a + "rope.dimension_count")
                || !targetSections.SequenceEqual(draftSections))
                throw new InvalidOperationException("qwen4exp MTP rotary sections do not match the target.");
            var compression = draft.GetInt32Array(a + "attention.compress_ratios");
            if (compression == null || compression.Length != layer + 1 || compression[layer] != 0)
                throw new NotSupportedException("qwen4exp MTP currently requires compression ratio zero on its draft block.");

            foreach (string key in new[] { "tokenizer.ggml.model", "tokenizer.ggml.pre" })
                if (target.GetString(key) == null || target.GetString(key) != draft.GetString(key))
                    throw new InvalidOperationException($"qwen4exp MTP tokenizer mismatch: {key}.");
            foreach (string key in new[] { "tokenizer.ggml.tokens", "tokenizer.ggml.merges" })
            {
                var x = target.GetStringArray(key);
                var y = draft.GetStringArray(key);
                if (x == null || y == null || !x.SequenceEqual(y, StringComparer.Ordinal))
                    throw new InvalidOperationException($"qwen4exp MTP token ID mapping mismatch: {key}.");
            }
            var targetTypes = target.GetInt32Array("tokenizer.ggml.token_type");
            var draftTypes = draft.GetInt32Array("tokenizer.ggml.token_type");
            if (targetTypes == null || draftTypes == null || !targetTypes.SequenceEqual(draftTypes))
                throw new InvalidOperationException("qwen4exp MTP token types do not match the target.");
            foreach (string key in new[] { "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id" })
                if (target.Metadata.ContainsKey(key) != draft.Metadata.ContainsKey(key)
                    || target.GetUint32(key) != draft.GetUint32(key))
                    throw new InvalidOperationException($"qwen4exp MTP special token mismatch: {key}.");
            if (target.GetBool("tokenizer.ggml.add_bos_token") != draft.GetBool("tokenizer.ggml.add_bos_token"))
                throw new InvalidOperationException("qwen4exp MTP BOS policy does not match the target.");

            ulong hidden = target.GetUint32(a + "embedding_length");
            ulong hc = target.GetUint32(a + "hyper_connection.count");
            ulong low = target.GetUint32(a + "hyper_connection.low_rank");
            ulong width = checked(hidden * hc);
            ulong heads = target.GetUint32(a + "attention.head_count");
            ulong kvHeads = target.GetUint32(a + "attention.head_count_kv");
            ulong dim = target.GetUint32(a + "attention.key_length");
            ulong experts = target.GetUint32(a + "expert_count");
            ulong ff = target.GetUint32(a + "expert_feed_forward_length");
            ulong shared = target.GetUint32(a + "expert_shared_feed_forward_length");
            ulong vocab = checked((ulong)target.GetStringArray("tokenizer.ggml.tokens").Length);
            if (hc < 2 || dim != draft.GetUint32(a + "attention.value_length")
                || heads % kvHeads != 0 || target.GetUint32(a + "expert_used_count") > experts
                || target.GetUint32(a + "rope.dimension_count") > dim
                || target.GetUint32(a + "rope.dimension_count") % 2 != 0 || vocab == 0)
                throw new NotSupportedException("qwen4exp MTP attention, HC, expert, or rotary dimensions are unsupported.");
            RequireTensor(targetTensor("token_embd.weight"), "token_embd.weight", false, hidden, vocab);
            RequireTensor(targetTensor("output.weight"), "output.weight", false, hidden, vocab);
            string p = $"blk.{layer}.";
            Require(draft, p + "nextn.enorm.weight", true, hidden);
            Require(draft, p + "nextn.hnorm.weight", true, width);
            Require(draft, p + "nextn.eh_proj.weight", false, checked(hidden * 2), hidden);
            Require(draft, p + "nextn.hc_head_norm.weight", true, width);
            Require(draft, p + "nextn.hc_head_down.weight", false, width, low);
            Require(draft, p + "nextn.hc_head_up.weight", false, low, width);
            foreach (string sub in new[] { "attn", "ffn" })
            {
                Require(draft, p + $"hc_{sub}_norm.weight", true, width);
                Require(draft, p + $"hc_{sub}_down.weight", false, width, low);
                Require(draft, p + $"hc_{sub}_up.weight", false, low, width);
                Require(draft, p + $"hc_{sub}_inject.weight", false, width, hc);
            }
            Require(draft, p + "attn_q.weight", false, hidden, checked(dim * heads * 2));
            Require(draft, p + "attn_k.weight", false, hidden, checked(dim * kvHeads));
            Require(draft, p + "attn_v.weight", false, hidden, checked(dim * kvHeads));
            Require(draft, p + "attn_output.weight", false, checked(dim * heads), hidden);
            Require(draft, p + "attn_q_norm.weight", true, dim);
            Require(draft, p + "attn_k_norm.weight", true, dim);
            Require(draft, p + "ffn_gate_inp.weight", false, hidden, experts);
            Require(draft, p + "ffn_gate_exps.weight", false, hidden, ff, experts);
            Require(draft, p + "ffn_up_exps.weight", false, hidden, ff, experts);
            Require(draft, p + "ffn_down_exps.weight", false, ff, hidden, experts);
            Require(draft, p + "ffn_gate_inp_shexp.weight", true, hidden);
            Require(draft, p + "ffn_gate_shexp.weight", false, hidden, shared);
            Require(draft, p + "ffn_up_shexp.weight", false, hidden, shared);
            Require(draft, p + "ffn_down_shexp.weight", false, shared, hidden);
            ulong indexDim = target.GetUint32(a + "attention.indexer.key_length");
            ulong indexHeads = target.GetUint32(a + "attention.indexer.head_count");
            Require(draft, p + "indexer.k_norm.weight", true, indexDim);
            Require(draft, p + "indexer.q_norm.weight", true, indexDim);
            Require(draft, p + "indexer.k_proj.weight", false, hidden, indexDim);
            Require(draft, p + "indexer.q_proj.weight", false, hidden, checked(indexDim * indexHeads));
            if (draft.Tensors.Count != 32)
                throw new NotSupportedException("qwen4exp shared MTP tensor directory has unrecognized tensors.");
            return layer;

            static void Require(GgufFile file, string name, bool f32, params ulong[] shape)
                => RequireTensor(file.Tensors.TryGetValue(name, out var t) ? t : null, name, f32, shape);

            static void RequireTensor(GgufTensorInfo tensor, string name, bool f32, params ulong[] shape)
            {
                if (tensor == null || !tensor.Shape.SequenceEqual(shape))
                    throw new InvalidOperationException($"qwen4exp MTP tensor is missing or has the wrong shape: {name}.");
                if (f32 && tensor.Type != GgmlTensorType.F32)
                    throw new NotSupportedException($"qwen4exp MTP norm must be F32: {name}.");
                if (!f32 && tensor.Type is GgmlTensorType.I8 or GgmlTensorType.I16
                    or GgmlTensorType.I32 or GgmlTensorType.I64 or GgmlTensorType.F64
                    or GgmlTensorType.Q8_1 or GgmlTensorType.Q8_K)
                    throw new NotSupportedException($"qwen4exp MTP matrix storage is unsupported: {name}.");
                long block = GgufFile.GetBlockSize(tensor.Type);
                if (block <= 0 || shape[0] % (ulong)block != 0 || GgufFile.GetTypeSize(tensor.Type) <= 0)
                    throw new NotSupportedException($"qwen4exp MTP tensor storage is unsupported: {name}.");
            }
        }
    }
}
