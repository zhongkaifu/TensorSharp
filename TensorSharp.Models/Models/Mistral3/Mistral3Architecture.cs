// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>Mistral 3 architecture plug-in.</summary>
    internal static class Mistral3Architecture
    {
        /// <summary>The only generic label a Mistral 3 checkpoint is known to carry.</summary>
        internal const string LlamaLabel = "llama";

        /// <summary>Control tokens <c>ChatTemplate.RenderMistral3</c> emits. A vocabulary
        /// without them would tokenize the framing as plain text.</summary>
        private static readonly string[] RequiredControlTokens = { "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]" };

        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "mistral3",
            DisplayName = "Mistral 3",
            Aliases = new[] { "mistral3" },
            Factory = c => new Mistral3Model(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
            RecognizeRelabelledFile = IsLlamaLabelledMistral3,
            RelabelledFileDescription =
                "a 'llama'-labelled dense Mistral checkpoint with the Tekken tokenizer, the [INST]/[SYSTEM_PROMPT] " +
                "control tokens, and no rope_freqs, experts or Q/K norms - e.g. bartowski's Mistral-Small-3.1-24B-Instruct-2503",
            // Plain name first, then the published companion names (bartowski ships
            // mmproj-mistralai_Mistral-Small-3.1-...-f16.gguf beside the model).
            ProjectorFileHints = new[] { "mistral3-mmproj.gguf", "*mmproj*istral*.gguf" },
        };

        /// <summary>
        /// True when a GGUF labelled <c>general.architecture = llama</c> is a Mistral 3
        /// text tower that <see cref="Mistral3Model"/> runs exactly: llama.cpp converted
        /// Mistral Small 3.x before it had a <c>mistral3</c> architecture, and that graph
        /// (GQA, SwiGLU, RMSNorm, normal-style RoPE, no QK-norm) is the same one.
        ///
        /// Every check refuses a neighbour that shares the label but not the graph:
        /// the Tekken pre-tokenizer and Mistral's control tokens rule out Llama 2/3 and
        /// SentencePiece Mistral 7B / Mixtral; <c>rope_freqs.weight</c> is Llama 3.1's
        /// frequency scaling, which this model does not apply; experts and per-head Q/K
        /// norms are layers it does not have; and only the RoPE scaling types it
        /// implements are accepted.
        /// </summary>
        internal static bool IsLlamaLabelledMistral3(string declaredArchitecture, GgufFile gguf)
        {
            if (gguf == null || !string.Equals(declaredArchitecture, LlamaLabel, StringComparison.OrdinalIgnoreCase))
                return false;

            string prefix = declaredArchitecture;
            if (!string.Equals(gguf.GetString("tokenizer.ggml.pre"), "tekken", StringComparison.Ordinal))
                return false;

            string[] tokens = gguf.GetStringArray("tokenizer.ggml.tokens");
            if (tokens == null || RequiredControlTokens.Any(t => Array.IndexOf(tokens, t) < 0))
                return false;

            if (gguf.GetUint32($"{prefix}.expert_count", 0) > 0)
                return false;

            string ropeScaling = gguf.GetString($"{prefix}.rope.scaling.type", "") ?? "";
            if (ropeScaling.Length > 0 && ropeScaling != "none" && ropeScaling != "yarn")
                return false;

            var tensors = gguf.Tensors;
            if (tensors.ContainsKey("rope_freqs.weight"))
                return false;
            foreach (string required in new[]
                     {
                         "token_embd.weight", "output_norm.weight",
                         "blk.0.attn_norm.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
                         "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                         "blk.0.ffn_norm.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                     })
            {
                if (!tensors.ContainsKey(required))
                    return false;
            }
            if (tensors.ContainsKey("blk.0.attn_q_norm.weight") || tensors.ContainsKey("blk.0.attn_k_norm.weight") ||
                tensors.ContainsKey("blk.0.ffn_gate_inp.weight"))
            {
                return false;
            }

            return gguf.GetUint32($"{prefix}.block_count", 0) > 0;
        }
    }
}
