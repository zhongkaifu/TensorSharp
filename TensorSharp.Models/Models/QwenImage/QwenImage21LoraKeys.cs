// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models.QwenImage;

/// <summary>What one LoRA file tensor contributes to a Qwen-Image-2.1 module.</summary>
internal enum QwenImage21LoraPart
{
    /// <summary>lora_A / lora_down: [rank, in].</summary>
    Down,
    /// <summary>lora_B / lora_up: [out, rank].</summary>
    Up,
    /// <summary>A one-element alpha (scale = alpha / rank).</summary>
    Alpha,
    /// <summary>DoRA magnitude, [out, 1].</summary>
    DoraScale,
    /// <summary>A full-weight difference (LyCORIS "diff"); supported for the 1-D norm gains.</summary>
    Diff,
    /// <summary>A plain ".weight": a full replacement value (a PDD bundle's norms and heads).</summary>
    Weight,
}

/// <summary>A LoRA file key resolved to the transformer module it targets.</summary>
internal readonly record struct QwenImage21LoraKey(string Module, QwenImage21LoraPart Part);

/// <summary>
/// Maps LoRA tensor names from every producer seen for Qwen-Image-2.1 onto the module
/// names of the transformer GGUF (without its <c>model.diffusion_model.</c> prefix):
/// diffusers/PEFT (<c>transformer.</c>, <c>lora_A/lora_B</c>), PEFT adapters with a slot
/// name (<c>lora_A.default.weight</c>), ComfyUI/ai-toolkit (<c>diffusion_model.</c>),
/// DiffSynth/ModelScope (no prefix), VideoX-Fun PDD bundles (bare <c>lora_down</c>),
/// kohya (<c>lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight</c>) and
/// <c>lora.down</c>/<c>lora.up</c> spellings.
/// </summary>
/// <remarks>
/// Diffusers' separate <c>img_mlp.gate_layer</c> and <c>img_mlp.proj</c> are kept as
/// such: they are the gate and the up half of the checkpoint's fused
/// <c>img_mlp.gate_up</c> (gate first), and the loader places them there.
/// Kohya names are matched against the known module list rather than by replacing
/// underscores, because the module names themselves contain underscores.
/// </remarks>
internal static class QwenImage21LoraKeys
{
    internal const int Layers = QwenImage21DiT.Layers;

    internal static readonly string[] GlobalModules =
    {
        "img_in", "txt_in.in_layer", "txt_in.out_layer",
        "time_text_embed.timestep_embedder.linear_1", "time_text_embed.timestep_embedder.linear_2",
        "modulation.1", "norm_out.linear", "proj_out",
    };

    internal static readonly string[] BlockModules =
    {
        "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
        "img_mlp.gate_up", "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out",
    };

    /// <summary>1-D gains a PDD bundle may replace (full value) or a diff may shift.</summary>
    internal static readonly string[] GainModules = { "txt_in.text_norm", "attn.norm_q", "attn.norm_k" };

    private static readonly string[] Prefixes =
    {
        "base_model.model.", "model.diffusion_model.", "diffusion_model.", "transformer.", "pipe.dit.", "dit.",
    };

    private static readonly HashSet<string> Known = BuildKnown();
    private static readonly Dictionary<string, string> Underscored = Known.ToDictionary(m => m.Replace('.', '_'), m => m, StringComparer.Ordinal);

    private static HashSet<string> BuildKnown()
    {
        var known = new HashSet<string>(StringComparer.Ordinal);
        foreach (var m in GlobalModules) known.Add(m);
        known.Add("txt_in.text_norm");
        for (int i = 0; i < Layers; i++)
        {
            foreach (var m in BlockModules) known.Add($"transformer_blocks.{i}.{m}");
            known.Add($"transformer_blocks.{i}.attn.norm_q");
            known.Add($"transformer_blocks.{i}.attn.norm_k");
        }
        return known;
    }

    internal static bool IsKnownModule(string module) => Known.Contains(module);

    /// <summary>
    /// Resolves <paramref name="key"/>, or returns null with <paramref name="reason"/> set.
    /// A key the loader cannot use is never skipped silently; the caller reports every one.
    /// </summary>
    internal static QwenImage21LoraKey? Resolve(string key, out string reason, out string slot)
    {
        reason = null;
        slot = null;
        string name = key;
        QwenImage21LoraPart? part = null;

        // Adapter formats TensorSharp does not implement fail with their own names.
        foreach (var marker in new[] { "lokr_", "hada_", ".lora_mid", ".lora_te", "lora_te_", "lora_te1_", "lora_te2_" })
            if (name.Contains(marker, StringComparison.Ordinal))
            {
                reason = marker.Contains("lora_te", StringComparison.Ordinal)
                    ? "text-encoder LoRA weights (Qwen-Image-2.1 LoRAs adapt only the transformer)"
                    : $"'{marker.Trim('.', '_')}' factors (LoKr / LoHa / LoCon-mid adapters are not supported)";
                return null;
            }

        // A PEFT adapter slot first (lora_A.<slot>.weight), before the plain ".weight" suffix.
        foreach (var (marker, p) in new[] { (".lora_A.", QwenImage21LoraPart.Down), (".lora_B.", QwenImage21LoraPart.Up) })
        {
            int at = name.LastIndexOf(marker, StringComparison.Ordinal);
            if (at > 0 && name.EndsWith(".weight", StringComparison.Ordinal) && name.Length - ".weight".Length > at + marker.Length)
            {
                string rest = name.Substring(at + marker.Length, name.Length - at - marker.Length - ".weight".Length);
                if (!rest.Contains('.'))
                {
                    slot = rest;
                    name = name.Substring(0, at);
                    part = p;
                    break;
                }
            }
        }
        // Then the factor suffix.
        (string Suffix, QwenImage21LoraPart Part)[] suffixes =
        {
            (".lora_A.weight", QwenImage21LoraPart.Down), (".lora_B.weight", QwenImage21LoraPart.Up),
            (".lora_down.weight", QwenImage21LoraPart.Down), (".lora_up.weight", QwenImage21LoraPart.Up),
            (".lora.down.weight", QwenImage21LoraPart.Down), (".lora.up.weight", QwenImage21LoraPart.Up),
            (".lora_down", QwenImage21LoraPart.Down), (".lora_up", QwenImage21LoraPart.Up),
            (".lora_A", QwenImage21LoraPart.Down), (".lora_B", QwenImage21LoraPart.Up),
            (".alpha", QwenImage21LoraPart.Alpha), (".dora_scale", QwenImage21LoraPart.DoraScale),
            (".diff", QwenImage21LoraPart.Diff), (".weight", QwenImage21LoraPart.Weight),
        };
        if (part == null)
            foreach (var (suffix, p) in suffixes)
            {
                if (name.EndsWith(suffix, StringComparison.Ordinal))
                {
                    name = name.Substring(0, name.Length - suffix.Length);
                    part = p;
                    break;
                }
            }
        if (part == null)
        {
            reason = name.EndsWith(".diff_b", StringComparison.Ordinal) || name.EndsWith(".bias", StringComparison.Ordinal)
                ? "a bias term (the Qwen-Image-2.1 transformer has no biases)"
                : "an unrecognized tensor kind";
            return null;
        }

        // Module path: drop wrapper prefixes, then map kohya underscore names.
        bool stripped = true;
        while (stripped)
        {
            stripped = false;
            foreach (var prefix in Prefixes)
                if (name.StartsWith(prefix, StringComparison.Ordinal))
                {
                    name = name.Substring(prefix.Length);
                    stripped = true;
                }
        }
        foreach (var kohya in new[] { "lora_unet_", "lycoris_", "lora_transformer_" })
            if (name.StartsWith(kohya, StringComparison.Ordinal))
            {
                string body = name.Substring(kohya.Length);
                if (Underscored.TryGetValue(body, out var mapped)) name = mapped;
                break;
            }
        if (!Known.Contains(name))
        {
            reason = "a module that the Qwen-Image-2.1 transformer does not have" +
                (name.Contains("add_q_proj", StringComparison.Ordinal) || name.Contains("txt_mlp", StringComparison.Ordinal) ||
                 name.Contains("img_mod", StringComparison.Ordinal) || name.Contains("net.0.proj", StringComparison.Ordinal)
                    ? " (a dual-stream Qwen-Image / Qwen-Image-Edit module: this LoRA was made for a different model)"
                    : "");
            return null;
        }
        return new QwenImage21LoraKey(name, part.Value);
    }

    /// <summary>Block index of a module name, or -1 for a global module.</summary>
    internal static int BlockOf(string module, out string local)
    {
        const string prefix = "transformer_blocks.";
        if (!module.StartsWith(prefix, StringComparison.Ordinal))
        {
            local = module;
            return -1;
        }
        int dot = module.IndexOf('.', prefix.Length);
        local = module.Substring(dot + 1);
        return int.Parse(module.AsSpan(prefix.Length, dot - prefix.Length));
    }
}
