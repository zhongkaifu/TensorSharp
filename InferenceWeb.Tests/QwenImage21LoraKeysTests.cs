// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins <see cref="QwenImage21LoraKeys.Resolve"/>: every LoRA key spelling seen for
/// Qwen-Image-2.1 (diffusers/PEFT, PEFT adapter slots, ComfyUI/ai-toolkit, DiffSynth,
/// VideoX-Fun PDD, kohya underscore names) lands on the transformer GGUF's module name,
/// and every key the loader cannot use is refused with a reason, never skipped.
/// </summary>
public sealed class QwenImage21LoraKeysTests
{
    public static IEnumerable<object[]> AcceptedKeys() => new[]
    {
        // diffusers / PEFT
        new object[] { "transformer.transformer_blocks.7.attn.to_q.lora_A.weight", "transformer_blocks.7.attn.to_q", "Down" },
        new object[] { "transformer.transformer_blocks.7.attn.to_q.lora_B.weight", "transformer_blocks.7.attn.to_q", "Up" },
        new object[] { "base_model.model.transformer_blocks.1.attn.to_out.0.lora_B.weight", "transformer_blocks.1.attn.to_out.0", "Up" },
        new object[] { "transformer.transformer_blocks.0.img_mlp.gate_layer.lora_B.weight", "transformer_blocks.0.img_mlp.gate_layer", "Up" },
        new object[] { "transformer.transformer_blocks.0.img_mlp.proj.lora_A.weight", "transformer_blocks.0.img_mlp.proj", "Down" },
        new object[] { "transformer.transformer_blocks.31.img_mlp.out.lora_A.weight", "transformer_blocks.31.img_mlp.out", "Down" },
        new object[] { "transformer.txt_in.in_layer.lora_A.weight", "txt_in.in_layer", "Down" },
        new object[] { "transformer.txt_in.out_layer.lora_B.weight", "txt_in.out_layer", "Up" },
        new object[] { "transformer.norm_out.linear.lora_B.weight", "norm_out.linear", "Up" },
        new object[] { "transformer.time_text_embed.timestep_embedder.linear_2.lora_A.weight", "time_text_embed.timestep_embedder.linear_2", "Down" },
        // PEFT adapter with a slot name
        new object[] { "transformer_blocks.0.attn.to_k.lora_A.default.weight", "transformer_blocks.0.attn.to_k", "Down" },
        new object[] { "transformer_blocks.0.attn.to_k.lora_B.default.weight", "transformer_blocks.0.attn.to_k", "Up" },
        // ComfyUI / ai-toolkit
        new object[] { "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_B.weight", "transformer_blocks.0.img_mlp.gate_up", "Up" },
        new object[] { "diffusion_model.transformer_blocks.0.attn.to_q.dora_scale", "transformer_blocks.0.attn.to_q", "DoraScale" },
        new object[] { "diffusion_model.transformer_blocks.0.attn.to_q.alpha", "transformer_blocks.0.attn.to_q", "Alpha" },
        new object[] { "model.diffusion_model.transformer_blocks.2.attn.to_v.lora_down.weight", "transformer_blocks.2.attn.to_v", "Down" },
        new object[] { "diffusion_model.transformer_blocks.2.attn.to_v.lora_up.weight", "transformer_blocks.2.attn.to_v", "Up" },
        new object[] { "diffusion_model.transformer_blocks.2.attn.to_v.lora.down.weight", "transformer_blocks.2.attn.to_v", "Down" },
        new object[] { "diffusion_model.transformer_blocks.2.attn.to_v.lora.up.weight", "transformer_blocks.2.attn.to_v", "Up" },
        // DiffSynth / ModelScope (no prefix) and pipe.dit / dit wrappers
        new object[] { "transformer_blocks.4.attn.to_out.0.lora_A.weight", "transformer_blocks.4.attn.to_out.0", "Down" },
        new object[] { "pipe.dit.transformer_blocks.0.attn.to_v.lora_B.weight", "transformer_blocks.0.attn.to_v", "Up" },
        new object[] { "dit.img_in.lora_A.weight", "img_in", "Down" },
        new object[] { "transformer_blocks.0.attn.to_q.lora_A", "transformer_blocks.0.attn.to_q", "Down" },
        new object[] { "transformer_blocks.0.attn.to_q.lora_B", "transformer_blocks.0.attn.to_q", "Up" },
        // VideoX-Fun PDD bundle: bare lora_down / lora_up, full values, per-step heads
        new object[] { "transformer_blocks.0.attn.to_q.lora_down", "transformer_blocks.0.attn.to_q", "Down" },
        new object[] { "img_in.lora_up", "img_in", "Up" },
        new object[] { "modulation.1.lora_down", "modulation.1", "Down" },
        new object[] { "proj_out.weight", "proj_out", "Weight" },
        new object[] { "transformer_blocks.3.attn.norm_q.weight", "transformer_blocks.3.attn.norm_q", "Weight" },
        new object[] { "transformer_blocks.3.attn.norm_k.weight", "transformer_blocks.3.attn.norm_k", "Weight" },
        new object[] { "txt_in.text_norm.weight", "txt_in.text_norm", "Weight" },
        // LyCORIS full-weight difference of a norm gain
        new object[] { "diffusion_model.transformer_blocks.5.attn.norm_k.diff", "transformer_blocks.5.attn.norm_k", "Diff" },
        // kohya: underscore names matched against the module list (the modules contain underscores)
        new object[] { "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight", "transformer_blocks.0.attn.to_q", "Down" },
        new object[] { "lora_unet_transformer_blocks_12_img_mlp_gate_up.lora_up.weight", "transformer_blocks.12.img_mlp.gate_up", "Up" },
        new object[] { "lora_unet_transformer_blocks_1_attn_to_out_0.lora_down.weight", "transformer_blocks.1.attn.to_out.0", "Down" },
        new object[] { "lora_unet_transformer_blocks_30_img_mlp_out.alpha", "transformer_blocks.30.img_mlp.out", "Alpha" },
        new object[] { "lora_unet_time_text_embed_timestep_embedder_linear_1.alpha", "time_text_embed.timestep_embedder.linear_1", "Alpha" },
        new object[] { "lora_unet_time_text_embed_timestep_embedder_linear_1.lora_up.weight", "time_text_embed.timestep_embedder.linear_1", "Up" },
        new object[] { "lora_unet_txt_in_in_layer.lora_down.weight", "txt_in.in_layer", "Down" },
        new object[] { "lora_unet_norm_out_linear.lora_up.weight", "norm_out.linear", "Up" },
        new object[] { "lora_unet_transformer_blocks_3_attn_norm_q.diff", "transformer_blocks.3.attn.norm_q", "Diff" },
        new object[] { "lycoris_transformer_blocks_0_attn_to_k.lora_up.weight", "transformer_blocks.0.attn.to_k", "Up" },
        new object[] { "lora_transformer_transformer_blocks_0_img_mlp_gate_layer.lora_down.weight", "transformer_blocks.0.img_mlp.gate_layer", "Down" },
    };

    [Theory]
    [MemberData(nameof(AcceptedKeys))]
    public void Resolve_MapsEveryProducerSpellingOntoTheTransformerModule(string key, string module, string part)
    {
        var resolved = QwenImage21LoraKeys.Resolve(key, out string reason, out _);

        Assert.True(resolved.HasValue, $"{key} was refused: {reason}");
        Assert.Null(reason);
        Assert.Equal(module, resolved.Value.Module);
        Assert.Equal(part, resolved.Value.Part.ToString());
        Assert.True(QwenImage21LoraKeys.IsKnownModule(resolved.Value.Module));
    }

    [Theory]
    [InlineData("transformer_blocks.0.attn.to_k.lora_A.default.weight", "default")]
    [InlineData("base_model.model.transformer_blocks.0.attn.to_k.lora_B.my_style.weight", "my_style")]
    [InlineData("transformer.transformer_blocks.0.attn.to_k.lora_A.weight", null)]
    [InlineData("diffusion_model.transformer_blocks.0.attn.to_k.lora_down.weight", null)]
    public void Resolve_ReportsThePeftAdapterSlot(string key, string slot)
    {
        var resolved = QwenImage21LoraKeys.Resolve(key, out _, out string actual);

        Assert.True(resolved.HasValue);
        Assert.Equal("transformer_blocks.0.attn.to_k", resolved.Value.Module);
        Assert.Equal(slot, actual);
    }

    [Theory]
    // LoKr / LoHa / LoCon-mid adapters
    [InlineData("lora_unet_transformer_blocks_0_attn_to_q.lokr_w1", "LoKr")]
    [InlineData("transformer_blocks.0.attn.to_q.hada_w1_a", "LoHa")]
    [InlineData("transformer_blocks.0.attn.to_q.lora_mid.weight", "LoCon-mid")]
    // text-encoder LoRAs
    [InlineData("lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight", "text-encoder")]
    [InlineData("lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_up.weight", "text-encoder")]
    [InlineData("text_encoder.lora_te.layers.0.q_proj.lora_A.weight", "text-encoder")]
    // biases (the transformer has none)
    [InlineData("transformer_blocks.0.attn.to_q.bias", "bias")]
    [InlineData("diffusion_model.transformer_blocks.0.attn.to_q.diff_b", "bias")]
    // dual-stream Qwen-Image / Qwen-Image-Edit modules
    [InlineData("transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight", "dual-stream")]
    [InlineData("transformer.transformer_blocks.0.txt_mlp.net.2.lora_B.weight", "dual-stream")]
    [InlineData("transformer.transformer_blocks.0.img_mod.1.lora_A.weight", "dual-stream")]
    [InlineData("transformer.transformer_blocks.0.img_mlp.net.0.proj.lora_A.weight", "dual-stream")]
    [InlineData("lora_unet_transformer_blocks_0_attn_add_q_proj.lora_down.weight", "dual-stream")]
    // modules the transformer does not have
    [InlineData("transformer.transformer_blocks.0.attn.to_x.lora_A.weight", "does not have")]
    [InlineData("transformer.transformer_blocks.32.attn.to_q.lora_A.weight", "does not have")]
    [InlineData("transformer_blocks.99.img_mlp.out.lora_B.weight", "does not have")]
    [InlineData("lora_unet_transformer_blocks_32_attn_to_q.lora_down.weight", "does not have")]
    [InlineData("lora_unet_mystery_module.lora_up.weight", "does not have")]
    [InlineData("transformer.pos_embed.lora_A.weight", "does not have")]
    // tensor kinds that are neither factors nor values
    [InlineData("transformer_blocks.0.attn.to_q.scale", "unrecognized tensor kind")]
    [InlineData("transformer_blocks.0.attn.to_q.lora_A.default", "unrecognized tensor kind")]
    public void Resolve_RefusesWithAHelpfulReason(string key, string reasonFragment)
    {
        var resolved = QwenImage21LoraKeys.Resolve(key, out string reason, out _);

        Assert.Null(resolved);
        Assert.False(string.IsNullOrWhiteSpace(reason));
        Assert.Contains(reasonFragment, reason, StringComparison.Ordinal);
    }

    [Fact]
    public void Resolve_DoesNotBlameADifferentModelForAnOrdinaryUnknownModule()
    {
        QwenImage21LoraKeys.Resolve("transformer.transformer_blocks.0.attn.to_x.lora_A.weight", out string reason, out _);

        Assert.DoesNotContain("dual-stream", reason, StringComparison.Ordinal);
    }

    [Fact]
    public void KnownModules_CoverEveryBlockAndNoMore()
    {
        foreach (var local in QwenImage21LoraKeys.BlockModules.Concat(new[] { "attn.norm_q", "attn.norm_k" }))
        {
            Assert.True(QwenImage21LoraKeys.IsKnownModule($"transformer_blocks.0.{local}"), local);
            Assert.True(QwenImage21LoraKeys.IsKnownModule($"transformer_blocks.{QwenImage21LoraKeys.Layers - 1}.{local}"), local);
            Assert.False(QwenImage21LoraKeys.IsKnownModule($"transformer_blocks.{QwenImage21LoraKeys.Layers}.{local}"), local);
        }
        foreach (var global in QwenImage21LoraKeys.GlobalModules)
            Assert.True(QwenImage21LoraKeys.IsKnownModule(global), global);
        Assert.True(QwenImage21LoraKeys.IsKnownModule("txt_in.text_norm"));
        Assert.False(QwenImage21LoraKeys.IsKnownModule("transformer_blocks.0.attn.add_q_proj"));
        Assert.Equal(32, QwenImage21LoraKeys.Layers);
    }

    [Theory]
    [InlineData("transformer_blocks.0.attn.to_q", 0, "attn.to_q")]
    [InlineData("transformer_blocks.12.img_mlp.gate_up", 12, "img_mlp.gate_up")]
    [InlineData("transformer_blocks.31.attn.to_out.0", 31, "attn.to_out.0")]
    [InlineData("transformer_blocks.7.attn.norm_k", 7, "attn.norm_k")]
    [InlineData("img_in", -1, "img_in")]
    [InlineData("time_text_embed.timestep_embedder.linear_1", -1, "time_text_embed.timestep_embedder.linear_1")]
    [InlineData("txt_in.text_norm", -1, "txt_in.text_norm")]
    public void BlockOf_SplitsTheBlockIndexFromTheLocalName(string module, int block, string local)
    {
        Assert.Equal(block, QwenImage21LoraKeys.BlockOf(module, out string actual));
        Assert.Equal(local, actual);
    }
}
