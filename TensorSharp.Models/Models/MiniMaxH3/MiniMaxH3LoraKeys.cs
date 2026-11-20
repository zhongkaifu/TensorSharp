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
//
// MiniMax-H3-Image's loader (MiniMaxH3ImageLoraSet) reads all three grammars of
// docs/design/diffusion-runtime.md section 5.3 through ResolveAny: diffusers PEFT (with diffusers'
// optional transformer. prefix), ComfyUI's native names (the checkpoint's own:
// [diffusion_model.]blocks.N.attn.qkv_proj / out_proj / mlp.fc1 / fc2, lora_A/B or lora_down/up, an
// optional .alpha) and kohya's (lora_unet_blocks_N_attn_qkv_proj...). The native grammars are in the
// checkpoint's layout already: qkv_proj one fused term over the stacked rows, fc1 in [gate; value].
// MiniMax-H3 video keeps Resolve, its diffusers-only grammar, unchanged.
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
        /// <summary>attn.qkv_proj as one projection (all 3 inner rows, stacked [q; k; v]): the ComfyUI and kohya grammars'
        /// fused term. Past <see cref="MiniMaxH3LoraKeys.TargetCount"/>: the diffusers grammar never yields it.</summary>
        Qkv,
    }

    /// <summary>The key grammars a MiniMax-H3 LoRA file is written in (docs/design/diffusion-runtime.md section 5.3).</summary>
    internal enum MiniMaxH3LoraGrammar
    {
        /// <summary>diffusers PEFT: <c>[transformer.]transformer_blocks.N.attn.to_q.lora_A[.slot].weight</c>,
        /// <c>token_refiner.refiner_blocks.N</c>; q, k and v apart; ff.net.0.proj's halves in [value; gate].</summary>
        Diffusers,
        /// <summary>ComfyUI's native names: <c>[diffusion_model.]blocks.N.attn.qkv_proj.lora_A.weight</c> (or
        /// <c>lora_down</c>/<c>lora_up</c>), <c>token_refiner.blocks.N</c>, an optional per-module <c>.alpha</c>; the
        /// checkpoint's own layout (fused qkv, fc1 in [gate; value]).</summary>
        Comfy,
        /// <summary>kohya: <c>lora_unet_blocks_N_attn_qkv_proj.lora_down.weight</c>,
        /// <c>lora_unet_token_refiner_blocks_N_...</c>, per-module <c>.alpha</c>; the checkpoint's layout.</summary>
        Kohya,
    }

    /// <summary>What a tensor of a LoRA file is to its term.</summary>
    internal enum MiniMaxH3LoraRole
    {
        /// <summary>lora_A / lora_down: [rank, in].</summary>
        Down,
        /// <summary>lora_B / lora_up: [out, rank].</summary>
        Up,
        /// <summary>The module's alpha, a scalar (ComfyUI and kohya).</summary>
        Alpha,
    }

    /// <summary>A tensor name of any grammar resolved to its block, target and role.</summary>
    /// <param name="Slot">The PEFT adapter name (diffusers only), or null.</param>
    internal readonly record struct MiniMaxH3LoraName(MiniMaxH3LoraGrammar Grammar, bool Refiner, int Block, MiniMaxH3LoraTarget Target,
        MiniMaxH3LoraRole Role, string Slot);

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
        /// <summary>The diffusers grammar's six targets (Q..Fc2); <see cref="MiniMaxH3LoraTarget.Qkv"/> is past them.</summary>
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

        // ---- the three grammars (MiniMax-H3-Image) ---------------------------------------------------------------

        /// <summary>diffusers PEFT, with diffusers' optional <c>transformer.</c> prefix.</summary>
        private static readonly Regex DiffusersPattern = new(
            @"^(?:transformer\.)?(?<scope>transformer_blocks|token_refiner\.refiner_blocks)\.(?<block>[0-9]+)\." +
            @"(?<module>attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)" +
            @"\.lora_(?<part>A|B)(\.(?<slot>[A-Za-z0-9_]+))?\.weight$",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        /// <summary>ComfyUI's native names (the checkpoint's), with or without <c>diffusion_model.</c>.</summary>
        private static readonly Regex ComfyPattern = new(
            @"^(?:diffusion_model\.)?(?<scope>blocks|token_refiner\.blocks)\.(?<block>[0-9]+)\." +
            @"(?<module>attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2)" +
            @"\.(?:(?<part>lora_A|lora_B|lora_down|lora_up)\.weight|(?<alpha>alpha))$",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        /// <summary>kohya's flattened names.</summary>
        private static readonly Regex KohyaPattern = new(
            @"^lora_unet_(?<scope>blocks|token_refiner_blocks)_(?<block>[0-9]+)_" +
            @"(?<module>attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)" +
            @"\.(?:(?<part>lora_down|lora_up)\.weight|(?<alpha>alpha))$",
            RegexOptions.CultureInvariant | RegexOptions.Compiled);

        /// <summary>The grammar a name is written in, judged by its prefix alone (null: none of the three), so a file that
        /// mixes grammars is refused by name even where a key does not resolve.</summary>
        internal static MiniMaxH3LoraGrammar? GrammarOf(string key)
        {
            ArgumentNullException.ThrowIfNull(key);
            if (key.StartsWith("lora_unet_", StringComparison.Ordinal)) return MiniMaxH3LoraGrammar.Kohya;
            if (key.StartsWith("diffusion_model.", StringComparison.Ordinal) || key.StartsWith("blocks.", StringComparison.Ordinal) ||
                key.StartsWith("token_refiner.blocks.", StringComparison.Ordinal))
                return MiniMaxH3LoraGrammar.Comfy;
            if (key.StartsWith("transformer.", StringComparison.Ordinal) || key.StartsWith(ScopeTrunk + ".", StringComparison.Ordinal) ||
                key.StartsWith(ScopeRefiner + ".", StringComparison.Ordinal))
                return MiniMaxH3LoraGrammar.Diffusers;
            return null;
        }

        /// <summary>The spelling of a grammar in configs and messages: <c>diffusers</c>, <c>comfy</c>, <c>kohya</c>.</summary>
        internal static string Spell(MiniMaxH3LoraGrammar grammar) => grammar switch
        {
            MiniMaxH3LoraGrammar.Diffusers => "diffusers",
            MiniMaxH3LoraGrammar.Comfy => "comfy",
            _ => "kohya",
        };

        /// <summary>
        /// Resolve a name of any of the three grammars for a transformer of <paramref name="blocks"/> trunk and
        /// <paramref name="refinerBlocks"/> token-refiner blocks, or return null with <paramref name="reason"/> set (the name
        /// then is not a target of this checkpoint, or not a LoRA factor at all).
        /// </summary>
        internal static MiniMaxH3LoraName? ResolveAny(string key, int blocks, int refinerBlocks, out string reason)
        {
            ArgumentNullException.ThrowIfNull(key);
            reason = null;
            MiniMaxH3LoraGrammar? grammar = GrammarOf(key);
            Match m = grammar switch
            {
                MiniMaxH3LoraGrammar.Diffusers => DiffusersPattern.Match(key),
                MiniMaxH3LoraGrammar.Comfy => ComfyPattern.Match(key),
                MiniMaxH3LoraGrammar.Kohya => KohyaPattern.Match(key),
                _ => Match.Empty,
            };
            if (grammar == null || !m.Success)
            {
                reason = AnyRejectionReason(key, grammar, blocks, refinerBlocks);
                return null;
            }
            string scope = m.Groups["scope"].Value;
            bool refiner = scope != "transformer_blocks" && scope != "blocks";
            int limit = refiner ? refinerBlocks : blocks;
            string index = m.Groups["block"].Value;
            if (index.Length > 1 && index[0] == '0')
            {
                reason = $"block index '{index}' has a leading zero, which no module name has";
                return null;
            }
            if (!int.TryParse(index, NumberStyles.None, CultureInfo.InvariantCulture, out int block) || block >= limit)
            {
                reason = OutOfRange(refiner, index, limit);
                return null;
            }
            var target = m.Groups["module"].Value switch
            {
                "attn.to_q" => MiniMaxH3LoraTarget.Q,
                "attn.to_k" => MiniMaxH3LoraTarget.K,
                "attn.to_v" => MiniMaxH3LoraTarget.V,
                "attn.to_out.0" or "attn.out_proj" or "attn_out_proj" => MiniMaxH3LoraTarget.Out,
                "ff.net.0.proj" or "mlp.fc1" or "mlp_fc1" => MiniMaxH3LoraTarget.Fc1,
                "ff.net.2" or "mlp.fc2" or "mlp_fc2" => MiniMaxH3LoraTarget.Fc2,
                _ => MiniMaxH3LoraTarget.Qkv,
            };
            MiniMaxH3LoraRole role = m.Groups["alpha"].Success ? MiniMaxH3LoraRole.Alpha
                : m.Groups["part"].Value is "B" or "lora_B" or "lora_up" ? MiniMaxH3LoraRole.Up : MiniMaxH3LoraRole.Down;
            return new MiniMaxH3LoraName(grammar.Value, refiner, block, target, role,
                m.Groups["slot"].Success ? m.Groups["slot"].Value : null);
        }

        /// <summary>Why a name is not a MiniMax-H3 LoRA term in its grammar.</summary>
        private static string AnyRejectionReason(string key, MiniMaxH3LoraGrammar? grammar, int blocks, int refinerBlocks)
        {
            if (grammar == null)
                return "an unsupported key prefix; MiniMax-H3 LoRA names start with transformer_blocks. or token_refiner.refiner_blocks. " +
                       "(diffusers, optionally under transformer.), diffusion_model.blocks. (ComfyUI) or lora_unet_blocks_ (kohya)";
            string kind = UnsupportedKind(key);
            if (kind != null) return kind;
            if (grammar == MiniMaxH3LoraGrammar.Diffusers)
            {
                string bare = key.StartsWith("transformer.", StringComparison.Ordinal) ? key.Substring("transformer.".Length) : key;
                return RejectionReason(bare, blocks, refinerBlocks);
            }
            // ComfyUI / kohya: a module of the block that is not a projection, or a tensor that is not a factor or alpha.
            string module = grammar == MiniMaxH3LoraGrammar.Kohya
                ? key.Substring("lora_unet_".Length).Split('.')[0]
                : key.Split(".lora_")[0];
            if (!key.EndsWith(".weight", StringComparison.Ordinal) && !key.EndsWith(".alpha", StringComparison.Ordinal))
                return "an unrecognized tensor kind (only lora_A / lora_B / lora_down / lora_up .weight factors and .alpha are accepted)";
            var scoped = grammar == MiniMaxH3LoraGrammar.Kohya
                ? Regex.Match(module, @"^(?<scope>blocks|token_refiner_blocks)_(?<block>[0-9]+)_(?<rest>.+)$", RegexOptions.CultureInvariant)
                : Regex.Match(module, @"^(?:diffusion_model\.)?(?<scope>blocks|token_refiner\.blocks)\.(?<block>[0-9]+)\.(?<rest>.+)$",
                    RegexOptions.CultureInvariant);
            if (!scoped.Success)
                return IsNonTarget(module) ? $"{NonTargetWhat(module)} is not a LoRA target" : $"'{module}' is not a LoRA target";
            bool refiner = scoped.Groups["scope"].Value != "blocks";
            string index = scoped.Groups["block"].Value;
            if (!int.TryParse(index, NumberStyles.None, CultureInfo.InvariantCulture, out int block) || block >= (refiner ? refinerBlocks : blocks))
                return OutOfRange(refiner, index, refiner ? refinerBlocks : blocks);
            string rest = scoped.Groups["rest"].Value;
            if (grammar == MiniMaxH3LoraGrammar.Kohya && rest is "attn_qkv_proj" or "attn_out_proj" or "mlp_fc1" or "mlp_fc2" ||
                grammar == MiniMaxH3LoraGrammar.Comfy && rest is "attn.qkv_proj" or "attn.out_proj" or "mlp.fc1" or "mlp.fc2")
                return $"an unrecognized tensor kind for {module} ({(grammar == MiniMaxH3LoraGrammar.Kohya ? "kohya takes lora_down / lora_up .weight" : "ComfyUI takes lora_A / lora_B / lora_down / lora_up .weight")} and .alpha)";
            return IsNonTarget(rest) ? $"{NonTargetWhat(rest)} is not a LoRA target" : $"'{rest}' is not a LoRA target";
        }

        /// <summary>Adapter kinds other than plain LoRA factors, by their tensor names.</summary>
        private static string UnsupportedKind(string key)
        {
            if (key.Contains("dora_scale", StringComparison.Ordinal) || key.Contains("lora_magnitude_vector", StringComparison.Ordinal))
                return "a DoRA magnitude, which TensorSharp does not apply (plain LoRA only)";
            if (key.Contains(".lora_mid.", StringComparison.Ordinal))
                return "a LoCon mid factor, which TensorSharp does not apply (plain LoRA only)";
            if (key.Contains(".hada_", StringComparison.Ordinal))
                return "a LoHa factor, which TensorSharp does not apply (plain LoRA only)";
            if (key.Contains(".lokr_", StringComparison.Ordinal))
                return "a LoKr factor, which TensorSharp does not apply (plain LoRA only)";
            if (key.EndsWith(".diff", StringComparison.Ordinal) || key.EndsWith(".diff_b", StringComparison.Ordinal))
                return "a full-weight difference, which TensorSharp does not apply (low-rank factors only)";
            if (key.EndsWith(".bias", StringComparison.Ordinal) || key.Contains(".lora_B.bias", StringComparison.Ordinal))
                return "a bias update, which TensorSharp does not apply (low-rank factors only)";
            return null;
        }

        /// <summary>The diffusers module name of a term, e.g. <c>transformer_blocks.10.ff.net.0.proj</c>.</summary>
        internal static string LoraModule(bool refiner, int block, MiniMaxH3LoraTarget target) =>
            $"{(refiner ? ScopeRefiner : ScopeTrunk)}.{block}." + target switch
            {
                MiniMaxH3LoraTarget.Q => "attn.to_q",
                MiniMaxH3LoraTarget.K => "attn.to_k",
                MiniMaxH3LoraTarget.V => "attn.to_v",
                MiniMaxH3LoraTarget.Out => "attn.to_out.0",
                MiniMaxH3LoraTarget.Fc1 => "ff.net.0.proj",
                MiniMaxH3LoraTarget.Qkv => "attn.qkv",
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
                MiniMaxH3LoraTarget.Qkv => $"{b}attn.qkv_proj rows [0, {3 * inner})",
                _ => b + "mlp.fc2",
            };
        }

        /// <summary>The input and output width of a target's update.</summary>
        internal static (long In, long Out) Shape(MiniMaxH3Config cfg, MiniMaxH3LoraTarget target) => target switch
        {
            MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V => (cfg.HiddenSize, cfg.AttentionInnerDim),
            MiniMaxH3LoraTarget.Out => (cfg.AttentionInnerDim, cfg.HiddenSize),
            MiniMaxH3LoraTarget.Fc1 => (cfg.HiddenSize, 2L * cfg.FfnHiddenSize),
            MiniMaxH3LoraTarget.Qkv => (cfg.HiddenSize, 3L * cfg.AttentionInnerDim),
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
