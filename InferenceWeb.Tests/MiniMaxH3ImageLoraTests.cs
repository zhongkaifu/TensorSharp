// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.SharedDiffusion;
using Xunit;
using G = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Grammar;
using P = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Proj;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image LoRA plug-ins (docs/design/diffusion-runtime.md section 5.3), without weights: the three key grammars
/// (<see cref="MiniMaxH3LoraKeys.ResolveAny"/>), the <c>minimax-h3-image-lora</c> config and its sampling contract, and the
/// loader (<see cref="MiniMaxH3ImageLoraSet"/>) on tiny generated files: every tensor used or named, a mix of grammars
/// refused, alpha by the section's rules, the update each grammar writes landing on the same checkpoint rows (q, k and v on
/// the stacked thirds, diffusers' fc1 halves swapped, a ComfyUI block-diagonal qkv split exactly), several style adapters
/// concatenated along the rank and two speed adapters refused.
/// </summary>
public sealed class MiniMaxH3ImageLoraTests : IDisposable
{
    private readonly MiniMaxH3ImageLoraFixture _fx = new();
    private readonly MiniMaxH3ImageFixture _dit = new(MiniMaxH3ImageFixture.F32);

    public void Dispose()
    {
        _fx.Dispose();
        _dit.Dispose();
    }

    private MiniMaxH3Config Config => MiniMaxH3Config.Detect(new MiniMaxH3WeightFile(_dit.DitPath).Tensors);

    private MiniMaxH3ImageLoraSet Load(BackendType backend, params LoraSpec[] specs) =>
        MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), Config, backend);

    // ---- the grammars --------------------------------------------------------------------------------------------------

    public static IEnumerable<object[]> GrammarNames() => new[]
    {
        new object[] { "transformer_blocks.1.attn.to_q.lora_A.weight", "Diffusers", false, 1, "Q", "Down" },
        new object[] { "transformer.transformer_blocks.0.attn.to_v.lora_B.default.weight", "Diffusers", false, 0, "V", "Up" },
        new object[] { "transformer.token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.weight", "Diffusers", true, 0, "Fc1", "Down" },
        new object[] { "transformer_blocks.1.attn.to_out.0.lora_B.weight", "Diffusers", false, 1, "Out", "Up" },
        new object[] { "diffusion_model.blocks.1.attn.qkv_proj.lora_A.weight", "Comfy", false, 1, "Qkv", "Down" },
        new object[] { "diffusion_model.blocks.0.mlp.fc1.lora_up.weight", "Comfy", false, 0, "Fc1", "Up" },
        new object[] { "diffusion_model.token_refiner.blocks.0.attn.out_proj.alpha", "Comfy", true, 0, "Out", "Alpha" },
        new object[] { "blocks.0.mlp.fc2.lora_B.weight", "Comfy", false, 0, "Fc2", "Up" },
        new object[] { "lora_unet_blocks_1_attn_qkv_proj.lora_down.weight", "Kohya", false, 1, "Qkv", "Down" },
        new object[] { "lora_unet_token_refiner_blocks_0_mlp_fc2.lora_up.weight", "Kohya", true, 0, "Fc2", "Up" },
        new object[] { "lora_unet_blocks_0_attn_out_proj.alpha", "Kohya", false, 0, "Out", "Alpha" },
    };

    [Theory]
    [MemberData(nameof(GrammarNames))]
    public void EachGrammarsNamesResolveToTheirBlockTargetAndRole(string key, string grammar, bool refiner, int block, string target, string role)
    {
        var name = MiniMaxH3LoraKeys.ResolveAny(key, 2, 1, out string reason);
        Assert.True(name.HasValue, $"{key}: {reason}");
        Assert.Equal(Enum.Parse<MiniMaxH3LoraGrammar>(grammar), name.Value.Grammar);
        Assert.Equal(refiner, name.Value.Refiner);
        Assert.Equal(block, name.Value.Block);
        Assert.Equal(Enum.Parse<MiniMaxH3LoraTarget>(target), name.Value.Target);
        Assert.Equal(Enum.Parse<MiniMaxH3LoraRole>(role), name.Value.Role);
    }

    public static IEnumerable<object[]> RefusedNames() => new[]
    {
        new object[] { "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight", "an unsupported key prefix" },
        new object[] { "transformer_blocks.0.attn.to_q.alpha", "a per-module .alpha tensor" },
        new object[] { "transformer_blocks.2.attn.to_q.lora_A.weight", "block 2 is out of range" },
        new object[] { "diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight", "(an AdaLN modulation parameter) is not a LoRA target" },
        new object[] { "diffusion_model.blocks.0.attn.qkv_proj.dora_scale", "a DoRA magnitude" },
        new object[] { "diffusion_model.blocks.0.attn.qkv_proj.lora_B.bias", "a bias update" },
        new object[] { "diffusion_model.blocks.0.attn.qkv_proj.diff", "a full-weight difference" },
        new object[] { "diffusion_model.token_refiner.blocks.1.mlp.fc1.lora_A.weight", "refiner block 1 is out of range" },
        new object[] { "diffusion_model.blocks.0.attn.to_q.lora_A.weight", "'attn.to_q' is not a LoRA target" },
        new object[] { "lora_unet_blocks_0_attn_qkv_proj.lora_A.weight", "kohya takes lora_down / lora_up" },
        new object[] { "lora_unet_blocks_0_attn_qkv_proj.hada_w1_a", "a LoHa factor" },
        new object[] { "lora_unet_blocks_00_mlp_fc1.lora_down.weight", "leading zero" },
        new object[] { "lora_te1_text_model_encoder.lora_down.weight", "an unsupported key prefix" },
    };

    [Theory]
    [MemberData(nameof(RefusedNames))]
    public void EveryOtherNameComesBackWithItsReason(string key, string reason)
    {
        Assert.Null(MiniMaxH3LoraKeys.ResolveAny(key, 2, 1, out string actual));
        Assert.Contains(reason, actual);
    }

    [Fact]
    public void TheVideoFamilysGrammarIsUnchanged()
    {
        // MiniMax-H3 video keeps its diffusers-only names (its Turbo LoRA tests pin the rest).
        Assert.Null(MiniMaxH3LoraKeys.Resolve("diffusion_model.blocks.0.mlp.fc1.lora_B.weight", 2, 1, out string comfy));
        Assert.Contains("a checkpoint-format (ComfyUI) name", comfy);
        Assert.Null(MiniMaxH3LoraKeys.Resolve("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", 2, 1, out _));
        Assert.NotNull(MiniMaxH3LoraKeys.Resolve("transformer_blocks.0.attn.to_q.lora_A.default.weight", 2, 1, out _));
    }

    // ---- the config ----------------------------------------------------------------------------------------------------

    [Fact]
    public void ASpeedConfigIsItsContract()
    {
        string weights = _fx.Write(MiniMaxH3ImageLoraFixture.Build(MiniMaxH3ImageLoraFixture.Random(1, 4, 1f, true), G.Diffusers), "pdmd.safetensors");
        var c = MiniMaxH3ImageLoraConfig.Load(_fx.WriteConfig("pdmd.json", weights, name: "Test pDMD", steps: 2));
        Assert.Equal("Test pDMD", c.Name);
        Assert.Equal(MiniMaxH3ImageLoraKeyFormat.Auto, c.KeyFormat);
        Assert.Equal("pdmd.safetensors", c.WeightsFileName);
        Assert.NotNull(c.Contract);
        Assert.Equal(2, c.Contract.Steps);
        Assert.Equal(new[] { 1f, 12f / 13f, 0f }, c.Contract.Sigmas);
        Assert.Equal(HeadPolicy.ByStep, c.Contract.Heads);
        Assert.Equal(1f, c.Contract.Cfg);
        Assert.Equal(1024, c.Contract.MinGenerationArea);
        var style = MiniMaxH3ImageLoraConfig.Load(_fx.WriteConfig("style.json", weights, keyFormat: "kohya", scale: 0.7f, alpha: 16));
        Assert.Null(style.Contract);
        Assert.Equal(MiniMaxH3ImageLoraKeyFormat.Kohya, style.KeyFormat);
        Assert.Equal(0.7f, style.Scale);
        Assert.Equal(16f, style.Alpha);
    }

    public static IEnumerable<object[]> BadConfigs() => new[]
    {
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "extra": 1 }""", "unknown key \"extra\"" },
        new object[] { """{ "type": "minimax-h3-lora", "name": "x", "weights": "w.safetensors" }""", "must be \"minimax-h3-image-lora\"" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "key_format": "peft" }""", "\"key_format\" must be" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "scale": -1 }""", "\"scale\" must be a strength >= 0" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "config": "p.json" }""", "forwards none" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5], "heads": "by-step", "cfg": 1, "min_generation_area": 1024 } }""", "steps + 1 = 3 values" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5, 0.1], "heads": "by-step", "cfg": 1, "min_generation_area": 1024 } }""", "must end in 0" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [0.5, 0.7, 0], "heads": "by-step", "cfg": 1, "min_generation_area": 1024 } }""", "must decrease" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5, 0], "heads": "per-step", "cfg": 1, "min_generation_area": 1024 } }""", "\"sampling.heads\" must be" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5, 0], "heads": "by-step", "cfg": 3.5, "min_generation_area": 1024 } }""", "guidance-distilled" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5, 0], "heads": "by-step", "cfg": 1, "min_generation_area": 100 } }""", "from 1024" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.safetensors", "sampling": { "steps": 2, "sigmas": [1, 0.5, 0], "heads": "by-step", "cfg": 1 } }""", "\"sampling.min_generation_area\" is required" },
        new object[] { """{ "type": "minimax-h3-image-lora", "name": "x", "weights": "w.gguf" }""", "must name a .safetensors file" },
    };

    [Theory]
    [MemberData(nameof(BadConfigs))]
    public void AMalformedConfigIsRefusedNamingItsKey(string json, string reason)
    {
        string path = Path.Combine(_fx.Dir, "bad.json");
        File.WriteAllText(path, json);
        var e = Assert.ThrowsAny<Exception>(() => MiniMaxH3ImageLoraConfig.Load(path));
        Assert.True(e is InvalidDataException or ArgumentException, e.ToString());
        Assert.Contains(reason, e.Message);
    }

    [Fact]
    public void TheHostsCheckTheirFlagsAgainstTheSpeedContractFromTheConfigsAlone()
    {
        // The weights need not exist yet: the host checks before any download.
        string speed = _fx.WriteConfig("speed.json", "pdmd.safetensors", name: "Speed", steps: 2);
        string style = _fx.WriteConfig("style.json", "style.safetensors", name: "Style");
        Assert.Null(MiniMaxH3ImageLoraConfig.CheckHostSettings(new[] { new LoraSpec(style) }, 6, "--diffusion-steps", 0, "--cfg"));
        Assert.Equal(2, MiniMaxH3ImageLoraConfig.CheckHostSettings(new[] { new LoraSpec(style), new LoraSpec(speed) }, 2, "--diffusion-steps", 1, "--cfg").Steps);
        Assert.Contains("--diffusion-steps 6 contradicts Speed",
            Assert.Throws<ArgumentException>(() => MiniMaxH3ImageLoraConfig.CheckHostSettings(new[] { new LoraSpec(speed) }, 6, "--diffusion-steps", 0, "--cfg")).Message);
        string speed2 = _fx.WriteConfig("speed2.json", "other.safetensors", name: "Speed 2", steps: 4, sigmas: new[] { 1f, 0.75f, 0.5f, 0.25f, 0f });
        Assert.Contains("Two speed adapters (Speed, Speed 2)",
            Assert.Throws<ArgumentException>(() => MiniMaxH3ImageLoraConfig.ReadConfigs(new[] { new LoraSpec(speed), new LoraSpec(speed2) })).Message);
        string video = Path.Combine(_fx.Dir, "video.json");
        File.WriteAllText(video, """{ "type": "minimax-h3-lora", "name": "v", "weights": "v.safetensors" }""");
        Assert.Contains("takes minimax-h3-image-lora plug-ins",
            Assert.Throws<ArgumentException>(() => MiniMaxH3ImageLoraConfig.ReadConfigs(new[] { new LoraSpec(video) })).Message);
        Assert.Contains("strength -1 is negative",
            Assert.Throws<ArgumentException>(() => MiniMaxH3ImageLoraConfig.ReadConfigs(new[] { new LoraSpec(speed, -1f) })).Message);
    }

    // ---- the loader ------------------------------------------------------------------------------------------------

    /// <summary>The delta a set holds for every projection of <paramref name="terms"/>, against the independent merge.</summary>
    private static void AssertDeltas(MiniMaxH3ImageLoraSet set, IReadOnlyList<(IReadOnlyList<MiniMaxH3ImageLoraFixture.Term>, float)> adapters, double bound)
    {
        var want = MiniMaxH3ImageLoraFixture.Deltas(adapters);
        int checkedProjections = 0;
        foreach (bool refiner in new[] { true, false })
            for (int b = 0; b < (refiner ? MiniMaxH3ImageLoraFixture.RefinerBlocks : MiniMaxH3ImageLoraFixture.Blocks); b++)
                foreach (P p in Enum.GetValues<P>())
                {
                    string name = MiniMaxH3ImageLoraFixture.WeightName(refiner, b, p);
                    var f = set.Get(refiner, b, (MiniMaxH3ImageLoraSet.Projection)(int)p);
                    if (!want.TryGetValue(name, out float[] expected))
                    {
                        Assert.Null(f);
                        continue;
                    }
                    var (input, output) = MiniMaxH3ImageLoraFixture.Shape(p);
                    double[] got = set.DeltaRows(refiner, b, (MiniMaxH3ImageLoraSet.Projection)(int)p, 0, output);
                    double err = 0, norm = 0;
                    for (long i = 0; i < expected.LongLength; i++)
                    {
                        err = Math.Max(err, Math.Abs(got[i] - expected[i]));
                        norm = Math.Max(norm, Math.Abs(expected[i]));
                    }
                    Assert.True(err <= bound * norm, $"{name}: max |delta - merge| {err:G4} > {bound} x {norm:G4}");
                    checkedProjections++;
                }
        Assert.True(checkedProjections > 0);
    }

    public static IEnumerable<object[]> Grammars() => new[]
    {
        new object[] { "Diffusers", true },
        new object[] { "Diffusers", false },
        new object[] { "Comfy", true },
        new object[] { "Comfy", false },
        new object[] { "Kohya", false },
        new object[] { "Kohya", true },
    };

    [Theory]
    [MemberData(nameof(Grammars))]
    public void EachGrammarPutsTheSameUpdateOnTheSameCheckpointRows(string grammar, bool perThirdQkv)
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(3, 4, 0.5f, perThirdQkv);
        string path = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, Enum.Parse<G>(grammar)), "a.safetensors");
        using var set = Load(BackendType.Cpu, new LoraSpec(path, 0.8f));
        Assert.Equal(Enum.Parse<MiniMaxH3LoraGrammar>(grammar), set.Adapters.Single().Grammar);
        // diffusers writes q, k and v apart: three terms where the others write one fused qkv term.
        Assert.Equal(terms.Sum(t => grammar == "Diffusers" && t.P == P.Qkv ? 3 : 1), set.Adapters.Single().Terms);
        AssertDeltas(set, new[] { ((IReadOnlyList<MiniMaxH3ImageLoraFixture.Term>)terms, 0.8f) }, 1e-6);
        // A ComfyUI block-diagonal qkv splits into its thirds (rank 4 each); a dense one stays one slot of all of qkv.
        var qkv = set.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Qkv);
        if (grammar == "Diffusers" || perThirdQkv)
        {
            Assert.Equal(MiniMaxH3ImageFixture.Inner, qkv.SlotRows);
            Assert.Equal(new[] { 0, 1, 2 }, qkv.Parts.Select(x => x.Slot));
            Assert.All(qkv.Parts, x => Assert.Equal(4, x.Rank));
        }
        else
        {
            Assert.Equal(0, qkv.SlotRows);
            Assert.Equal(4, Assert.Single(qkv.Parts).Rank);
        }
    }

    [Fact]
    public void SeveralAdaptersConcatenateAlongTheRankEachAtItsStrength()
    {
        var speed = MiniMaxH3ImageLoraFixture.Random(5, 8, 1f, true);
        var style = MiniMaxH3ImageLoraFixture.Random(6, 4, 1f, false, trunkOnly: true, projections: new[] { P.Qkv, P.Fc1 });
        var style2 = MiniMaxH3ImageLoraFixture.Random(7, 2, 0.5f, false, trunkOnly: true, projections: new[] { P.Out });
        string s = _fx.Write(MiniMaxH3ImageLoraFixture.Build(speed, G.Diffusers), "speed.safetensors");
        string t = _fx.Write(MiniMaxH3ImageLoraFixture.Build(style, G.Kohya), "style.safetensors");
        string u = _fx.Write(MiniMaxH3ImageLoraFixture.Build(style2, G.Comfy, downUp: true), "style2.safetensors");
        string speedConfig = _fx.WriteConfig("speed.json", s, name: "Speed", steps: 2);
        using var set = Load(BackendType.GgmlMetal, new LoraSpec(speedConfig), new LoraSpec(t, 0.6f), new LoraSpec(u, 1.5f));
        Assert.Equal(new[] { "Speed", "style", "style2" }, set.Adapters.Select(a => a.Name));
        Assert.Equal(2, set.Contract.Steps);
        Assert.Equal(3, set.AdapterSet.Adapters.Count);
        Assert.All(set.AdapterSet.Adapters, a => Assert.Same(set, a.Payload));
        Assert.Equal(new[] { 1f, 0.6f, 1.5f }, set.AdapterSet.Adapters.Select(a => a.Scale));
        // F16 factors on Metal, ranks padded to 64: qkv's thirds take 8 + 4 (the dense style term reaches every third).
        var qkv = set.Get(false, 1, MiniMaxH3ImageLoraSet.Projection.Qkv);
        Assert.All(qkv.Parts, x => Assert.Equal(GraphFormatF16, x.Format));
        Assert.All(qkv.Parts, x => Assert.Equal(64, x.Rank));
        AssertDeltas(set, new[]
        {
            ((IReadOnlyList<MiniMaxH3ImageLoraFixture.Term>)speed, 1f), (style, 0.6f), (style2, 1.5f),
        }, 4e-3);
        Assert.Contains("speed adapter", set.Summary);
        Assert.Contains("style adapter", set.Summary);
    }

    private const int GraphFormatF16 = TensorSharp.Graph.GraphFormat.F16;

    [Fact]
    public void GgmlCpuKeepsF32FactorsAndOtherGgmlBackendsPadTheRankTo16()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(8, 3, 1f, false, trunkOnly: true, projections: new[] { P.Fc2 });
        string path = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy), "c.safetensors");
        using var cpu = Load(BackendType.GgmlCpu, new LoraSpec(path));
        Assert.Equal(TensorSharp.Graph.GraphFormat.F32, cpu.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Fc2).Parts[0].Format);
        Assert.Equal(16, cpu.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Fc2).Parts[0].Rank);
        using var cuda = Load(BackendType.GgmlCuda, new LoraSpec(path));
        Assert.Equal(GraphFormatF16, cuda.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Fc2).Parts[0].Format);
        using var managed = Load(BackendType.Cpu, new LoraSpec(path));
        Assert.Equal(3, managed.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Fc2).Parts[0].Rank);
        Assert.Null(managed.Get(false, 0, MiniMaxH3ImageLoraSet.Projection.Out));
        Assert.Null(managed.Get(true, 0, MiniMaxH3ImageLoraSet.Projection.Fc2));
    }

    [Fact]
    public void AlphaFollowsEachGrammarsRule()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(9, 4, 0.25f, false, trunkOnly: true, projections: new[] { P.Out });
        // diffusers: the metadata's alpha (1 at rank 4); none: the config's, else the rank (scale 1).
        string withAlpha = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Diffusers), "d-alpha.safetensors");
        string noAlpha = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Diffusers, alpha: false), "d-none.safetensors");
        using (var set = Load(BackendType.Cpu, new LoraSpec(withAlpha)))
            Assert.Equal(new[] { 0.25f }, set.Adapters[0].Scales);
        using (var set = Load(BackendType.Cpu, new LoraSpec(noAlpha)))
        {
            Assert.Equal(new[] { 1f }, set.Adapters[0].Scales);
            Assert.Equal("rank", set.Adapters[0].AlphaSource);
        }
        using (var set = Load(BackendType.Cpu, new LoraSpec(_fx.WriteConfig("d.json", noAlpha, alpha: 2))))
            Assert.Equal(new[] { 0.5f }, set.Adapters[0].Scales);
        var conflict = Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(_fx.WriteConfig("d2.json", withAlpha, alpha: 2))));
        Assert.Contains("sets alpha 2, but the file's metadata says 1", conflict.Message);
        // ComfyUI / kohya: each module's .alpha tensor; none: the config's, else the rank.
        string comfy = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy), "c-alpha.safetensors");
        string comfyNone = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy, alpha: false), "c-none.safetensors");
        using (var set = Load(BackendType.Cpu, new LoraSpec(comfy)))
        {
            Assert.Equal(new[] { 0.25f }, set.Adapters[0].Scales);
            Assert.Equal(".alpha tensors", set.Adapters[0].AlphaSource);
        }
        using (var set = Load(BackendType.Cpu, new LoraSpec(comfyNone)))
            Assert.Equal(new[] { 1f }, set.Adapters[0].Scales);
        Assert.Contains("but c.json sets alpha 3",
            Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(_fx.WriteConfig("c.json", comfy, alpha: 3)))).Message);
    }

    [Fact]
    public void EveryTensorIsUsedOrTheLoadFailsNamingIt()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(10, 2, 1f, false, trunkOnly: true, projections: new[] { P.Fc2 });
        var file = MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy);
        file.Tensors.Add(("diffusion_model.blocks.0.adaln_proj.linear.lora_A.weight", new long[] { 2, 8 }, new float[16], "F32"));
        file.Tensors.Add(("diffusion_model.blocks.1.attn.out_proj.lora_A.weight", new long[] { 2, MiniMaxH3ImageFixture.Inner }, new float[2 * MiniMaxH3ImageFixture.Inner], "F32"));
        file.Tensors.Add(("diffusion_model.blocks.0.mlp.fc1.alpha", Array.Empty<long>(), new[] { 2f }, "F32"));
        string path = _fx.Write(file, "extra.safetensors");
        var e = Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(path)));
        Assert.Contains("3 problem(s)", e.Message);
        Assert.Contains("adaln_proj.linear.lora_A.weight: 'adaln_proj.linear' (an AdaLN modulation parameter) is not a LoRA target", e.Message);
        Assert.Contains("diffusion_model.blocks.1.attn.out_proj: has only its down factor", e.Message);
        Assert.Contains("diffusion_model.blocks.0.mlp.fc1.alpha: an .alpha without its module's factors", e.Message);
        Assert.Contains("Nothing is skipped silently", e.Message);
    }

    [Fact]
    public void AMixOfGrammarsIsRefusedNamingBoth()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(11, 2, 1f, false, trunkOnly: true, projections: new[] { P.Out });
        var file = MiniMaxH3ImageLoraFixture.Build(terms, G.Diffusers);
        file.Tensors.AddRange(MiniMaxH3ImageLoraFixture.Build(MiniMaxH3ImageLoraFixture.Random(12, 2, 1f, false, trunkOnly: true,
            projections: new[] { P.Fc2 }), G.Kohya).Tensors);
        var e = Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(_fx.Write(file, "mixed.safetensors"))));
        Assert.Contains("the file mixes key grammars (diffusers: transformer_blocks.0.attn.to_out.0.lora_A.weight, kohya: lora_unet_blocks_0_mlp_fc2.alpha)", e.Message);
    }

    [Fact]
    public void ADeclaredKeyFormatMustBeTheFilesAndTheShapesTheCheckpoints()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(13, 2, 1f, false, trunkOnly: true, projections: new[] { P.Out });
        string path = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Kohya), "k.safetensors");
        Assert.Contains("declares key_format \"comfy\", but the names are kohya",
            Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(_fx.WriteConfig("k.json", path, keyFormat: "comfy")))).Message);
        var wrong = new MiniMaxH3ImageLoraFixture.File();
        wrong.Tensors.Add(("lora_unet_blocks_0_mlp_fc2.lora_down.weight", new long[] { 2, 100 }, new float[200], "F32"));
        wrong.Tensors.Add(("lora_unet_blocks_0_mlp_fc2.lora_up.weight", new long[] { MiniMaxH3ImageFixture.Hidden, 2 }, new float[2 * MiniMaxH3ImageFixture.Hidden], "F32"));
        Assert.Contains("a LoRA made for a different model",
            Assert.Throws<InvalidDataException>(() => Load(BackendType.Cpu, new LoraSpec(_fx.Write(wrong, "w.safetensors")))).Message);
    }

    [Fact]
    public void TwoSpeedAdaptersAndAForeignConfigAreRefused()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(14, 2, 1f, false, trunkOnly: true, projections: new[] { P.Out });
        string a = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy), "a.safetensors");
        string b = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy), "b.safetensors");
        var e = Assert.Throws<ArgumentException>(() => Load(BackendType.Cpu,
            new LoraSpec(_fx.WriteConfig("a.json", a, name: "A", steps: 2)), new LoraSpec(_fx.WriteConfig("b.json", b, name: "B", steps: 2))));
        Assert.Contains("Two speed adapters (A, B) each fix the sampling schedule", e.Message);
        string peft = Path.Combine(_fx.Dir, "adapter_config.json");
        File.WriteAllText(peft, """{ "r": 2, "lora_alpha": 2 }""");
        Assert.Contains("reads its own minimax-h3-image-lora configs only",
            Assert.Throws<ArgumentException>(() => Load(BackendType.Cpu, new LoraSpec(a, null, peft))).Message);
        Assert.Contains("is the Test adapter plug-in for b.safetensors",
            Assert.Throws<ArgumentException>(() => Load(BackendType.Cpu, new LoraSpec(a, null, _fx.WriteConfig("tie.json", b)))).Message);
    }

    [Fact]
    public void DisposingASetDropsItsDeviceCopiesFirst()
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(15, 2, 1f, false, trunkOnly: true, projections: new[] { P.Out, P.Fc2 });
        string a = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Comfy), "a.safetensors");
        var seen = new List<IntPtr>();
        MiniMaxH3ImageLoraSet.InvalidateHookForTest = p => seen.Add(p);
        try
        {
            var set = Load(BackendType.GgmlMetal, new LoraSpec(a));
            var pointers = set.FactorPointers().ToArray();
            Assert.Equal(2 * 2 * 2, pointers.Length);
            set.Dispose();
            // (Other tests may dispose sets meanwhile: only this set's pointers are counted.)
            Assert.Equal(pointers, seen.Where(pointers.Contains));
            set.Dispose();
            Assert.Equal(pointers.Length, seen.Count(pointers.Contains));
        }
        finally
        {
            MiniMaxH3ImageLoraSet.InvalidateHookForTest = null;
        }
    }
}
