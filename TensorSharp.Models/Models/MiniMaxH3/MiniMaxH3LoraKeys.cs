// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 LoRA tensor names ("key_format": "minimax-h3-diffusers") and where each one
// lands in the GGUF transformer.
//
// The adapters (lightx2v/Minimax-h3-Turbo *_bf16.safetensors) are PEFT saves of diffusers'
// MiniMaxH3 transformer: transformer_blocks.N.attn.to_q.lora_A.default.weight and so on. The
// GGUF checkpoints are the single-file layout, and diffusers' converter
// (src/diffusers/loaders/single_file_utils.py:4319-4371,
// convert_minimax_h3_transformer_checkpoint_to_diffusers) is the authority for the mapping:
//   * blocks.N.attn.qkv_proj is STACKED [q; k; v] and splits into to_q / to_k / to_v by
//     thirds, so a to_q/to_k/to_v update lands on qkv_proj rows [0, i), [i, 2i), [2i, 3i)
//     (i = heads x head_dim), before the q/k RMSNorm and RoPE;
//   * mlp.fc1 fuses [gate; value] while diffusers' SwiGLU reads ff.net.0.proj as
//     [value; gate] (convert_minimax_h3_gated_ff: "The checkpoint fuses [gate; value],
//     SwiGLU reads [value; gate]"), so lora_B's two row halves are swapped into fc1's order;
//   * attn.to_out.0 is out_proj and ff.net.2 is fc2, with no input permutation;
//   * token_refiner.refiner_blocks.N is the GGUF's token_refiner.blocks.N.
// The publisher's ComfyUI twin of each adapter is converted the same way; its metadata reads
// swi_glu_mapping 'Diffusers [value;gate] -> ComfyUI [gate;value]' and qkv_fusion 'block
// diagonal B; concat A; alpha multiplied by 3'. InferenceWeb.Tests holds rows of that twin
// (Fixtures/MiniMaxH3/turbo-comfy-reference-rows.json) as an independent check of both facts.
using System;
using System.Globalization;
using System.Text.RegularExpressions;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The projection of a MiniMax-H3 DiT or token-refiner block a LoRA term updates.</summary>
    internal enum MiniMaxH3LoraTarget
    {
        /// <summary>attn.to_q: qkv_proj rows [0, inner).</summary>
        Q,
        /// <summary>attn.to_k: qkv_proj rows [inner, 2 inner).</summary>
        K,
        /// <summary>attn.to_v: qkv_proj rows [2 inner, 3 inner).</summary>
        V,
        /// <summary>attn.to_out.0: out_proj.</summary>
        Out,
        /// <summary>ff.net.0.proj: mlp.fc1, lora_B halves swapped to [gate | value].</summary>
        Fc1,
        /// <summary>ff.net.2: mlp.fc2.</summary>
        Fc2,
    }

    /// <summary>One LoRA tensor resolved to its block, target and factor.</summary>
    /// <param name="Up">True for lora_B ([out, rank]), false for lora_A ([rank, in]).</param>
    /// <param name="Slot">The PEFT adapter name (<c>default</c>), or null when the key has none.</param>
    internal readonly record struct MiniMaxH3LoraKey(bool Refiner, int Block, MiniMaxH3LoraTarget Target, bool Up, string Slot);

    /// <summary>
    /// Resolves <c>minimax-h3-diffusers</c> LoRA tensor names. Every name either resolves or
    /// comes back with the reason it cannot apply; the loader reports all of them.
    /// </summary>
    internal static class MiniMaxH3LoraKeys
    {
        internal const int TargetCount = 6;

        /// <summary>The whole accepted key grammar: two scopes x six modules x two factors, with an
        /// optional PEFT slot.</summary>
        internal static readonly Regex Pattern = new(
            @"^(?<scope>transformer_blocks|token_refiner\.refiner_blocks)\.(?<block>0|[1-9][0-9]*)\." +
            @"(?<module>attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)" +
            @"\.lora_(?<part>A|B)(\.(?<slot>[A-Za-z0-9_]+))?\.weight$",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        /// <summary>Any factor-shaped key, whatever its module path, to tell a non-target module
        /// apart from an unrecognized tensor.</summary>
        private static readonly Regex FactorKey = new(
            @"^(?<path>.+?)\.lora_(A|B)(\.[A-Za-z0-9_]+)?\.weight$", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        private static readonly Regex BlockScope = new(
            @"^(?<scope>transformer_blocks|token_refiner\.refiner_blocks)\.(?<block>[0-9]+)\.(?<rest>.+)$",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        internal const string ScopeTrunk = "transformer_blocks";
        internal const string ScopeRefiner = "token_refiner.refiner_blocks";

        /// <summary>
        /// Resolve <paramref name="key"/> for a transformer of <paramref name="blocks"/> trunk and
        /// <paramref name="refinerBlocks"/> token-refiner blocks, or return null with
        /// <paramref name="reason"/> set.
        /// </summary>
        internal static MiniMaxH3LoraKey? Resolve(string key, int blocks, int refinerBlocks, out string reason)
        {
            reason = null;
            var m = Pattern.Match(key);
            if (m.Success)
            {
                bool refiner = m.Groups["scope"].Value == ScopeRefiner;
                int limit = refiner ? refinerBlocks : blocks;
                if (!int.TryParse(m.Groups["block"].Value, NumberStyles.None, CultureInfo.InvariantCulture, out int block) ||
                    block >= limit)
                {
                    reason = OutOfRange(refiner, m.Groups["block"].Value, limit);
                    return null;
                }
                var target = m.Groups["module"].Value switch
                {
                    "attn.to_q" => MiniMaxH3LoraTarget.Q,
                    "attn.to_k" => MiniMaxH3LoraTarget.K,
                    "attn.to_v" => MiniMaxH3LoraTarget.V,
                    "attn.to_out.0" => MiniMaxH3LoraTarget.Out,
                    "ff.net.0.proj" => MiniMaxH3LoraTarget.Fc1,
                    _ => MiniMaxH3LoraTarget.Fc2,
                };
                return new MiniMaxH3LoraKey(refiner, block, target, m.Groups["part"].Value == "B",
                    m.Groups["slot"].Success ? m.Groups["slot"].Value : null);
            }
            reason = RejectionReason(key, blocks, refinerBlocks);
            return null;
        }

        /// <summary>Why <paramref name="key"/> is not a MiniMax-H3 LoRA target.</summary>
        private static string RejectionReason(string key, int blocks, int refinerBlocks)
        {
            if (key.EndsWith(".alpha", StringComparison.Ordinal))
                return "a per-module .alpha tensor; MiniMax-H3 plug-ins take alpha from the file's __metadata__ \"alpha\" " +
                       "or the plug-in config, never from tensors";
            var factor = FactorKey.Match(key);
            if (!factor.Success)
                return "an unrecognized tensor kind (only lora_A / lora_B .weight factors are accepted)";
            string path = factor.Groups["path"].Value;
            var scoped = BlockScope.Match(path);
            if (!scoped.Success)
                return path.StartsWith("diffusion_model.", StringComparison.Ordinal) || path.StartsWith("blocks.", StringComparison.Ordinal)
                    ? "an unsupported key prefix: a checkpoint-format (ComfyUI) name; minimax-h3-diffusers keys start with " +
                      $"{ScopeTrunk}. or {ScopeRefiner}."
                    : IsNonTarget(path)
                        ? $"{NonTargetWhat(path)} is not a LoRA target"
                        : $"an unsupported key prefix; minimax-h3-diffusers keys start with {ScopeTrunk}. or {ScopeRefiner}.";
            bool refiner = scoped.Groups["scope"].Value == ScopeRefiner;
            int limit = refiner ? refinerBlocks : blocks;
            string index = scoped.Groups["block"].Value;
            if (index.Length > 1 && index[0] == '0')
                return $"block index '{index}' has a leading zero, which no diffusers module name has";
            if (!int.TryParse(index, NumberStyles.None, CultureInfo.InvariantCulture, out int block) || block >= limit)
                return OutOfRange(refiner, index, limit);
            string rest = scoped.Groups["rest"].Value;
            return IsNonTarget(rest)
                ? $"{NonTargetWhat(rest)} is not a LoRA target"
                : $"'{rest}' is not a LoRA target";
        }

        private static bool IsNonTarget(string path) =>
            Contains(path, "adaln") || Contains(path, "norm") || Contains(path, "time") || Contains(path, "patch") ||
            Contains(path, "condition") || Contains(path, "final") || Contains(path, "proj_out");

        private static string NonTargetWhat(string path) =>
            Contains(path, "adaln") ? $"'{path}' (an AdaLN modulation parameter)"
            : Contains(path, "norm") ? $"'{path}' (a normalization parameter)"
            : Contains(path, "time") ? $"'{path}' (a timestep-embedding parameter)"
            : Contains(path, "patch") ? $"'{path}' (a patch-embedding parameter)"
            : Contains(path, "condition") ? $"'{path}' (a condition-projection parameter)"
            : $"'{path}' (a final-layer parameter)";

        private static bool Contains(string path, string word) => path.Contains(word, StringComparison.OrdinalIgnoreCase);

        private static string OutOfRange(bool refiner, string block, int limit) =>
            $"{(refiner ? "refiner block" : "block")} {block} is out of range (the checkpoint has {limit} " +
            $"{(refiner ? "token-refiner blocks" : "transformer blocks")})";

        /// <summary>The diffusers module name of a term, e.g. <c>transformer_blocks.10.ff.net.0.proj</c>.</summary>
        internal static string LoraModule(bool refiner, int block, MiniMaxH3LoraTarget target) =>
            $"{(refiner ? ScopeRefiner : ScopeTrunk)}.{block}." + target switch
            {
                MiniMaxH3LoraTarget.Q => "attn.to_q",
                MiniMaxH3LoraTarget.K => "attn.to_k",
                MiniMaxH3LoraTarget.V => "attn.to_v",
                MiniMaxH3LoraTarget.Out => "attn.to_out.0",
                MiniMaxH3LoraTarget.Fc1 => "ff.net.0.proj",
                _ => "ff.net.2",
            };

        /// <summary>The GGUF projection a term updates, e.g. <c>blocks.10.attn.qkv_proj rows [0, 7168)</c>.</summary>
        internal static string GgufTarget(bool refiner, int block, MiniMaxH3LoraTarget target, long inner)
        {
            string b = (refiner ? "token_refiner.blocks." : "blocks.") + block.ToString(CultureInfo.InvariantCulture) + ".";
            return target switch
            {
                MiniMaxH3LoraTarget.Q => $"{b}attn.qkv_proj rows [0, {inner})",
                MiniMaxH3LoraTarget.K => $"{b}attn.qkv_proj rows [{inner}, {2 * inner})",
                MiniMaxH3LoraTarget.V => $"{b}attn.qkv_proj rows [{2 * inner}, {3 * inner})",
                MiniMaxH3LoraTarget.Out => b + "attn.out_proj",
                MiniMaxH3LoraTarget.Fc1 => b + "mlp.fc1 (lora_B halves swapped to [gate | value])",
                _ => b + "mlp.fc2",
            };
        }

        /// <summary>The input and output width of a target's update.</summary>
        internal static (long In, long Out) Shape(MiniMaxH3Config cfg, MiniMaxH3LoraTarget target) => target switch
        {
            MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V => (cfg.HiddenSize, cfg.AttentionInnerDim),
            MiniMaxH3LoraTarget.Out => (cfg.AttentionInnerDim, cfg.HiddenSize),
            MiniMaxH3LoraTarget.Fc1 => (cfg.HiddenSize, 2L * cfg.FfnHiddenSize),
            _ => (cfg.FfnHiddenSize, cfg.HiddenSize),
        };

        /// <summary>
        /// Swap the two row halves of an up factor <paramref name="up"/> [<paramref name="output"/>,
        /// <paramref name="rank"/>] in place: diffusers' ff.net.0.proj [value; gate] becomes the
        /// checkpoint's fc1 [gate | value]. Its own inverse.
        /// </summary>
        internal static void SwapHalves(float[] up, long output, int rank)
        {
            if (output % 2 != 0)
                throw new ArgumentException($"An fc1 update has {output} output rows; the SwiGLU halves need an even count.", nameof(output));
            long half = output / 2 * rank;
            if (up.LongLength < 2 * half)
                throw new ArgumentException($"The up factor holds {up.LongLength} values, fewer than {output} x {rank}.", nameof(up));
            for (long i = 0; i < half; i++)
            {
                float t = up[i];
                up[i] = up[i + half];
                up[i + half] = t;
            }
        }
    }
}
