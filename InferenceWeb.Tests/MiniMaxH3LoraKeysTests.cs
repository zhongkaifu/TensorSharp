// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins <see cref="MiniMaxH3LoraKeys"/>: the twelve <c>minimax-h3-diffusers</c> module
/// patterns (two scopes x six projections) resolve to their block and target with either
/// factor and with or without a PEFT slot; every other name comes back with the reason it
/// cannot apply; the GGUF placement follows diffusers' single-file converter (stacked
/// [q; k; v] thirds, fc1 halves swapped).
/// </summary>
public sealed class MiniMaxH3LoraKeysTests
{
    private const int Blocks = 50, RefinerBlocks = 2;

    private static readonly (string Module, MiniMaxH3LoraTarget Target)[] Modules =
    {
        ("attn.to_q", MiniMaxH3LoraTarget.Q),
        ("attn.to_k", MiniMaxH3LoraTarget.K),
        ("attn.to_v", MiniMaxH3LoraTarget.V),
        ("attn.to_out.0", MiniMaxH3LoraTarget.Out),
        ("ff.net.0.proj", MiniMaxH3LoraTarget.Fc1),
        ("ff.net.2", MiniMaxH3LoraTarget.Fc2),
    };

    /// <summary>The published MiniMax-H3 DiT: 5376 wide, 56 x 128 attention, ffn 14336.</summary>
    private static readonly MiniMaxH3Config RealConfig = new()
    {
        HiddenSize = 5376,
        NumLayers = Blocks,
        TokenRefinerLayers = RefinerBlocks,
        NumAttentionHeads = 56,
        AttentionHeadDim = 128,
        FfnHiddenSize = 14336,
    };

    public static IEnumerable<object[]> AcceptedKeys()
    {
        foreach (var (scope, refiner, block) in new[] { ("transformer_blocks", false, 0), ("transformer_blocks", false, 49),
                     ("token_refiner.refiner_blocks", true, 0), ("token_refiner.refiner_blocks", true, 1) })
            foreach (var (module, target) in Modules)
                foreach (string part in new[] { "A", "B" })
                    foreach (string slot in new[] { "default", null })
                        yield return new object[]
                        {
                            $"{scope}.{block}.{module}.lora_{part}{(slot == null ? "" : "." + slot)}.weight",
                            refiner, block, target.ToString(), part == "B", slot,
                        };
    }

    [Theory]
    [MemberData(nameof(AcceptedKeys))]
    public void Resolve_MapsEachOfTheTwelvePatterns(string key, bool refiner, int block, string target, bool up, string slot)
    {
        var resolved = MiniMaxH3LoraKeys.Resolve(key, Blocks, RefinerBlocks, out string reason);

        Assert.True(resolved.HasValue, $"{key} was refused: {reason}");
        Assert.Null(reason);
        Assert.Equal(new MiniMaxH3LoraKey(refiner, block, Enum.Parse<MiniMaxH3LoraTarget>(target), up, slot), resolved.Value);
    }

    [Fact]
    public void LoraModule_SpellsTheKeysResolveAccepts()
    {
        foreach (bool refiner in new[] { false, true })
            foreach (var (module, target) in Modules)
            {
                string name = MiniMaxH3LoraKeys.LoraModule(refiner, 1, target);
                Assert.Equal((refiner ? "token_refiner.refiner_blocks.1." : "transformer_blocks.1.") + module, name);
                var resolved = MiniMaxH3LoraKeys.Resolve(name + ".lora_A.default.weight", Blocks, RefinerBlocks, out _);
                Assert.Equal(new MiniMaxH3LoraKey(refiner, 1, target, false, "default"), resolved);
            }
    }

    public static IEnumerable<object[]> RefusedKeys() => new[]
    {
        // Alpha comes from the metadata or the plug-in config, never from per-module tensors.
        new object[] { "transformer_blocks.0.attn.to_q.alpha", "a per-module .alpha tensor" },
        // Not a factor at all.
        new object[] { "transformer_blocks.0.attn.to_q.weight", "an unrecognized tensor kind" },
        new object[] { "transformer_blocks.0.attn.to_q.lora_down.weight", "an unrecognized tensor kind" },
        new object[] { "transformer_blocks.0.attn.to_q.lora_magnitude_vector.default.weight", "an unrecognized tensor kind" },
        // The ComfyUI twins' checkpoint-format names, and other prefixes.
        new object[] { "diffusion_model.blocks.10.mlp.fc1.lora_B.weight", "a checkpoint-format (ComfyUI) name" },
        new object[] { "blocks.10.attn.qkv_proj.lora_A.weight", "a checkpoint-format (ComfyUI) name" },
        new object[] { "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight", "an unsupported key prefix" },
        new object[] { "transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "an unsupported key prefix" },
        // Parameters that are not LoRA targets.
        new object[] { "transformer_blocks.3.adaln_proj.linear.lora_A.weight", "(an AdaLN modulation parameter) is not a LoRA target" },
        new object[] { "transformer_blocks.3.norm1.lora_A.weight", "(a normalization parameter) is not a LoRA target" },
        new object[] { "transformer_blocks.3.attn.q_norm.lora_B.weight", "(a normalization parameter) is not a LoRA target" },
        new object[] { "time_embedder.linear_1.lora_A.weight", "(a timestep-embedding parameter) is not a LoRA target" },
        new object[] { "video_patch_proj.lora_A.weight", "(a patch-embedding parameter) is not a LoRA target" },
        new object[] { "condition_proj.lora_B.weight", "(a condition-projection parameter) is not a LoRA target" },
        new object[] { "final_layer.video_out.lora_A.weight", "(a final-layer parameter) is not a LoRA target" },
        new object[] { "proj_out.lora_A.weight", "(a final-layer parameter) is not a LoRA target" },
        new object[] { "transformer_blocks.3.attn.to_add_out.lora_A.weight", "'attn.to_add_out' is not a LoRA target" },
        new object[] { "transformer_blocks.3.ff.net.0.lora_A.weight", "'ff.net.0' is not a LoRA target" },
        // Block indices.
        new object[] { "transformer_blocks.50.attn.to_q.lora_A.weight", "block 50 is out of range (the checkpoint has 50 transformer blocks)" },
        new object[] { "token_refiner.refiner_blocks.2.ff.net.2.lora_B.weight", "refiner block 2 is out of range (the checkpoint has 2 token-refiner blocks)" },
        new object[] { "transformer_blocks.07.attn.to_q.lora_A.weight", "block index '07' has a leading zero" },
        new object[] { "transformer_blocks.99999999999.attn.to_q.lora_A.weight", "block 99999999999 is out of range" },
    };

    [Theory]
    [MemberData(nameof(RefusedKeys))]
    public void Resolve_RefusesEveryOtherNameWithItsReason(string key, string reason)
    {
        var resolved = MiniMaxH3LoraKeys.Resolve(key, Blocks, RefinerBlocks, out string actual);

        Assert.Null(resolved);
        Assert.NotNull(actual);
        Assert.Contains(reason, actual);
    }

    [Theory]
    [InlineData("Q", 5376, 7168)]
    [InlineData("K", 5376, 7168)]
    [InlineData("V", 5376, 7168)]
    [InlineData("Out", 7168, 5376)]
    [InlineData("Fc1", 5376, 28672)]
    [InlineData("Fc2", 14336, 5376)]
    public void Shape_IsTheGgufProjectionsWidths(string target, long input, long output) =>
        Assert.Equal((input, output), MiniMaxH3LoraKeys.Shape(RealConfig, Enum.Parse<MiniMaxH3LoraTarget>(target)));

    [Theory]
    [InlineData(false, "Q", "blocks.10.attn.qkv_proj rows [0, 7168)")]
    [InlineData(false, "K", "blocks.10.attn.qkv_proj rows [7168, 14336)")]
    [InlineData(false, "V", "blocks.10.attn.qkv_proj rows [14336, 21504)")]
    [InlineData(false, "Out", "blocks.10.attn.out_proj")]
    [InlineData(false, "Fc1", "blocks.10.mlp.fc1 (lora_B halves swapped to [gate | value])")]
    [InlineData(true, "Fc2", "token_refiner.blocks.10.mlp.fc2")]
    public void GgufTarget_PlacesQkvInStackedThirds(bool refiner, string target, string expected) =>
        Assert.Equal(expected, MiniMaxH3LoraKeys.GgufTarget(refiner, 10, Enum.Parse<MiniMaxH3LoraTarget>(target), 7168));

    [Fact]
    public void SwapHalves_TurnsValueGateIntoGateValue_AndIsItsOwnInverse()
    {
        const int rank = 3, output = 4;
        // diffusers ff.net.0.proj rows: value 0, value 1, gate 0, gate 1.
        float[] up = { 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42 };
        float[] original = (float[])up.Clone();

        MiniMaxH3LoraKeys.SwapHalves(up, output, rank);
        Assert.Equal(new float[] { 30, 31, 32, 40, 41, 42, 10, 11, 12, 20, 21, 22 }, up);

        MiniMaxH3LoraKeys.SwapHalves(up, output, rank);
        Assert.Equal(original, up);
    }

    [Fact]
    public void SwapHalves_TouchesOnlyTheLeadingTermOfALargerBuffer()
    {
        float[] up = { 1, 2, 3, 4, 99, 98 };
        MiniMaxH3LoraKeys.SwapHalves(up, output: 4, rank: 1);
        Assert.Equal(new float[] { 3, 4, 1, 2, 99, 98 }, up);
    }

    [Fact]
    public void SwapHalves_RefusesAnOddRowCountAndAShortBuffer()
    {
        Assert.Contains("even count", Assert.Throws<ArgumentException>(() => MiniMaxH3LoraKeys.SwapHalves(new float[6], 3, 2)).Message);
        Assert.Contains("fewer than 4 x 2", Assert.Throws<ArgumentException>(() => MiniMaxH3LoraKeys.SwapHalves(new float[6], 4, 2)).Message);
    }
}
