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
using TensorSharp.Models;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using Xunit;
using G = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Grammar;
using P = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Proj;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image as an <see cref="ILoraPluginHost"/> (docs/design/diffusion-runtime.md section 5.3): what it accepts, how
/// <see cref="MiniMaxH3ImageDiffusionModel.SetLoras"/> swaps a set (the old one freed, its device copies dropped; a refused
/// set leaving the previous one), the speed contract a request may only repeat, and the hosts' startup check
/// (<see cref="LoraHostPreflight"/>): the model's own configs and bare weights taken, another family's refused, the CLI's
/// flags against the contract before any download, the safetensors release probed.
/// </summary>
public sealed class MiniMaxH3ImageLoraHostTests : IDisposable
{
    private readonly MiniMaxH3ImageLoraFixture _fx = new();
    private readonly MiniMaxH3ImageFixture _dit = new(MiniMaxH3ImageFixture.Q8_0);
    private readonly MiniMaxH3ImageVaeFixture _vae = new(2);

    public MiniMaxH3ImageLoraHostTests()
    {
        // The companions the model resolves beside its denoiser (only the text encoder's name is read at load).
        File.WriteAllBytes(Path.Combine(_dit.Dir, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"), new byte[16]);
        File.Copy(_vae.VaePath, Path.Combine(_dit.Dir, "minimax_h3_image_vae_fp16.safetensors"));
    }

    public void Dispose()
    {
        _fx.Dispose();
        _dit.Dispose();
        _vae.Dispose();
    }

    private string Speed(string name = "Speed", int steps = 2, string heads = "by-step", float[] sigmas = null)
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(51, 4, 1f, true, projections: new[] { P.Qkv, P.Fc2 });
        string weights = _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Diffusers), $"{name}.safetensors");
        return _fx.WriteConfig($"{name}.json", weights, name: name, steps: steps, heads: heads, sigmas: sigmas);
    }

    private string Style(string file = "style.safetensors")
    {
        var terms = MiniMaxH3ImageLoraFixture.Random(52, 2, 1f, false, trunkOnly: true, projections: new[] { P.Out });
        return _fx.Write(MiniMaxH3ImageLoraFixture.Build(terms, G.Kohya), file);
    }

    private MiniMaxH3ImageDiffusionModel Model() =>
        Assert.IsType<MiniMaxH3ImageDiffusionModel>(ModelBase.Create(_dit.DitPath, TestGates.PinnedGgmlBackend));

    [Fact]
    public void TheModelTakesItsOwnConfigsAndBareWeightsOnly()
    {
        using var model = Model();
        Assert.Equal("minimax-h3-image-lora", model.LoraPluginType);
        Assert.Empty(model.LoraSpecs);
        Assert.Null(model.SamplingContract);
        Assert.True(model.AcceptsLora(new LoraSpec(Speed())));
        Assert.True(model.AcceptsLora(new LoraSpec(Style())));
        string video = Path.Combine(_fx.Dir, "video.json");
        File.WriteAllText(video, """{ "type": "minimax-h3-lora", "name": "v", "weights": "v.safetensors" }""");
        string qwen = Path.Combine(_fx.Dir, "qwen.json");
        File.WriteAllText(qwen, """{ "type": "qwen-image-2.1-lora", "weights": "q.safetensors" }""");
        Assert.False(model.AcceptsLora(new LoraSpec(video)));
        Assert.False(model.AcceptsLora(new LoraSpec(qwen)));
        Assert.Throws<FileNotFoundException>(() => model.AcceptsLora(new LoraSpec(Path.Combine(_fx.Dir, "missing.json"))));
    }

    [Fact]
    public void SetLorasSwapsTheSetAndARefusedSetLeavesThePreviousOne()
    {
        using var model = Model();
        string speed = Speed(), style = Style();
        model.SetLoras(new[] { new LoraSpec(speed), new LoraSpec(style, 0.5f) });
        MiniMaxH3ImageLoraSet first = model.Loras;
        Assert.Equal(2, model.LoraSpecs.Count);
        Assert.Equal(Path.Combine(_fx.Dir, "Speed.safetensors"), model.LoraSpecs[0].Path);   // the plug-in expanded to its weights
        Assert.Equal(new[] { "Speed", "style" }, model.Adapters.Adapters.Select(a => a.Name));
        Assert.Equal(2, model.SamplingContract.Steps);
        Assert.Equal(HeadPolicy.ByStep, model.SamplingContract.Heads);

        // Two speed adapters: refused before anything changes.
        string other = Speed("Other");
        Assert.Contains("Two speed adapters (Speed, Other)",
            Assert.Throws<ArgumentException>(() => model.SetLoras(new[] { new LoraSpec(speed), new LoraSpec(other) })).Message);
        Assert.Same(first, model.Loras);

        // By-step heads need a head for every step: this checkpoint has 6.
        string seven = Speed("Seven", steps: 7, sigmas: new[] { 1f, 0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0f });
        Assert.Contains("this checkpoint has 6 heads", Assert.Throws<ArgumentException>(() => model.SetLoras(new[] { new LoraSpec(seven) })).Message);
        Assert.Same(first, model.Loras);

        // The swap frees the old set, its device copies first; an empty list removes the plug-ins.
        var seen = new List<IntPtr>();
        IntPtr[] firstPointers = first.FactorPointers().ToArray();
        MiniMaxH3ImageLoraSet.InvalidateHookForTest = p => seen.Add(p);
        try
        {
            model.SetLoras(new[] { new LoraSpec(style) });
            Assert.NotSame(first, model.Loras);
            Assert.Null(model.SamplingContract);
            Assert.Equal(firstPointers, seen.Where(firstPointers.Contains));
            model.SetLoras(Array.Empty<LoraSpec>());
            Assert.Null(model.Loras);
            Assert.True(model.Adapters.IsEmpty);
            Assert.Empty(model.LoraSpecs);
        }
        finally
        {
            MiniMaxH3ImageLoraSet.InvalidateHookForTest = null;
        }
    }

    [Fact]
    public void ARequestMayOnlyRepeatTheSpeedContract()
    {
        using var model = Model();
        model.SetLoras(new[] { new LoraSpec(Speed()) });
        // Refused in the pipeline's Prepare, before any component runs.
        Assert.Contains("Speed samples exactly 2 steps on its own schedule; 6 steps contradicts it",
            Assert.Throws<ArgumentException>(() => model.GenerateImage("x", new QwenImageParams { Steps = 6, Width = 64, Height = 64 })).Message);
        Assert.Contains("cfg must be 1",
            Assert.Throws<ArgumentException>(() => model.GenerateImage("x", new QwenImageParams { CfgScale = 4, Width = 64, Height = 64 })).Message);
        Assert.False(model.Pipeline.IsResident(DiffusionComponent.Denoiser));
    }

    // ---- the hosts' startup check --------------------------------------------------------------------------------------

    private static LoraHostVerdict Check(IReadOnlyList<LoraSpec> flags, string model, int steps = 0, float cfg = 0f, LoraHost host = LoraHost.Cli) =>
        LoraHostPreflight.Check(new LoraHostRequest(host, flags, Array.Empty<LoraSpec>(), model,
            host == LoraHost.Cli ? LoraHostPreflight.CliSampling(steps, 0f, cfg, null) : LoraHostPreflight.ServerSampling(0)));

    [Fact]
    public void TheHostsPreflightTakesTheModelsPluginsAndChecksTheContract()
    {
        string speed = Speed(), style = Style();
        LoraHostVerdict verdict = Check(new[] { new LoraSpec(speed), new LoraSpec(style) }, _dit.DitPath, steps: 2);
        Assert.Equal(MiniMaxH3ImageFamily.ArchitectureId, verdict.Architecture?.Id);
        Assert.Empty(verdict.EnvWarnings);
        Assert.Contains("--diffusion-steps 6 contradicts Speed, which samples exactly 2 steps",
            Assert.Throws<ArgumentException>(() => Check(new[] { new LoraSpec(speed) }, _dit.DitPath, steps: 6)).Message);
        // The server's requests name their own steps: checked per request, not at startup.
        Check(new[] { new LoraSpec(speed) }, _dit.DitPath, host: LoraHost.Server);
        string video = Path.Combine(_fx.Dir, "video.json");
        File.WriteAllText(video, """{ "type": "minimax-h3-lora", "name": "v", "weights": "v.safetensors" }""");
        Assert.Contains("takes minimax-h3-image-lora plug-ins only",
            Assert.Throws<ArgumentException>(() => Check(new[] { new LoraSpec(video) }, _dit.DitPath)).Message);
    }

    [Fact]
    public void TheSafetensorsReleaseIsProbedForThePreflight()
    {
        using var bf16 = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.BF16, safetensors: true);
        Assert.Equal(MiniMaxH3ImageFamily.ArchitectureId, Check(new[] { new LoraSpec(Style()) }, bf16.DitPath).Architecture?.Id);
    }
}
