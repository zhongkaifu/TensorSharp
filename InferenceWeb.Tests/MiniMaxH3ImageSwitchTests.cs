// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image is served by the shared diffusion runtime (stage S4 of docs/design/diffusion-runtime.md): the registry
/// claims the safetensors release through the family's detector and a GGUF denoiser through its tensors (the 'hyvid' label
/// included), both loads build the runtime's model, the contract is the runtime's, and the legacy model is a test seam only.
/// </summary>
public sealed class MiniMaxH3ImageSwitchTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-h3img-switch-" + Guid.NewGuid().ToString("N"));

    public MiniMaxH3ImageSwitchTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>The companions the model resolves beside its denoiser: a text encoder file (only its name is read at load)
    /// and the tiny VAE.</summary>
    private static void Companions(string dir, MiniMaxH3ImageVaeFixture vae)
    {
        File.WriteAllBytes(Path.Combine(dir, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"), new byte[16]);
        File.Copy(vae.VaePath, Path.Combine(dir, "minimax_h3_image_vae_fp16.safetensors"));
    }

    [Fact]
    public void TheDescriptorIsTheDiffusionRuntimesWithTheFamilysDetectors()
    {
        ModelArchitectureDescriptor d = ModelArchitectureRegistry.All.Single(a => a.Id == MiniMaxH3ImageFamily.ArchitectureId);
        Assert.Equal(ServingImplementation.SharedDiffusionRuntime, d.Serving.Implementation);
        Assert.Equal(new[] { BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda }, d.Serving.Backends);
        Assert.False(d.Serving.Autoregressive);
        Assert.Equal(SupportKind.NotApplicable, d.Serving.ContinuousBatching.Kind);
        Assert.Equal(ServingContract.DiffusionJobsReason, d.Serving.ContinuousBatching.Reason);
        Assert.NotNull(d.DetectSafetensors);
        Assert.NotNull(d.DetectFromTensors);
        Assert.NotNull(d.RecognizeRelabelledFile);
        Assert.Contains("minimax_h3_image", d.Aliases);
    }

    /// <summary>Stage S10 deleted the legacy model, its decoder and its denoiser seam: the runtime's model is the only one.</summary>
    [Fact]
    public void TheLegacyModelIsGone()
    {
        System.Reflection.Assembly models = typeof(ModelBase).Assembly;
        foreach (string gone in new[]
        {
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3ImageModel",
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3ImageVae",
            "TensorSharp.Models.MiniMaxH3Image.MiniMaxH3ImageLegacyDenoiser",
            // The audio-video DiT whose ForwardImage the image model had used is gone too (MiniMaxH3Denoiser replaced it).
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3DiT",
        })
            Assert.Null(models.GetType(gone));
        Assert.Null(typeof(MiniMaxH3Config).GetMethod("IsMiniMaxH3Image"));
        Assert.True(typeof(MiniMaxH3ImageDiffusionModel).IsPublic);
        Assert.True(typeof(IImageGenerationModel).IsAssignableFrom(typeof(MiniMaxH3ImageDiffusionModel)));
    }

    /// <summary>The audio-video denoiser refuses an image denoiser by name instead of reading its head bank as one head.</summary>
    [Fact]
    public void TheVideoDenoiserRefusesAnImageDenoiser()
    {
        using var dit = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.BF16);
        var ex = Assert.Throws<NotSupportedException>(() => new MiniMaxH3Denoiser(dit.DitPath, TestGates.PinnedGgmlBackend));
        Assert.Contains("MiniMax-H3-Image", ex.Message);
        Assert.Contains("minimax-h3-image architecture", ex.Message);
    }

    /// <summary>The safetensors release goes through the registry's safetensors hook (no special case in ModelBase.Create)
    /// and builds the runtime's model, with the checkpoint's grid and the companions beside it; nothing reaches a device
    /// before the first image.</summary>
    [PinnedGgmlServesFact(MiniMaxH3ImageFamily.ArchitectureId)]
    public void TheSafetensorsReleaseLoadsAsTheRuntimesModel()
    {
        using var dit = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.BF16, safetensors: true);
        using var vae = new MiniMaxH3ImageVaeFixture(2);
        Companions(dit.Dir, vae);
        Assert.Equal(MiniMaxH3ImageFamily.ArchitectureId, ModelArchitectureRegistry.ResolveSafetensors(dit.DitPath)?.Id);
        using ModelBase model = ModelBase.Create(dit.DitPath, TestGates.PinnedGgmlBackend);
        var runtime = Assert.IsType<MiniMaxH3ImageDiffusionModel>(model);
        Assert.True(runtime.SupportsImageEdit);
        Assert.Equal(MiniMaxH3ImageFamily.ArchitectureId, model.Config.Architecture);
        Assert.Equal(MiniMaxH3ImageSampling.GridFrom(new System.Collections.Generic.Dictionary<string, string>(), MiniMaxH3ImageFixture.PddSteps),
            runtime.PddSigmas.ToArray());
        Assert.Equal(MiniMaxH3ImageFixture.Blocks, runtime.DitConfig.NumLayers);
        Assert.False(runtime.Pipeline.IsResident(DiffusionComponent.Denoiser));
        Assert.Contains("at least one reference image", Assert.Throws<ArgumentException>(() =>
            runtime.EditImage("x", Array.Empty<RgbImage>(), new QwenImageParams())).Message);
        var picture = new RgbImage(64, 64, new float[64 * 64 * 3]);
        Assert.Contains("cfg must be 1", Assert.Throws<ArgumentException>(() =>
            runtime.EditImage("x", new[] { picture }, new QwenImageParams { CfgScale = 2 })).Message);
        Assert.Contains("negative prompt", Assert.Throws<ArgumentException>(() =>
            runtime.EditImage("x", new[] { picture }, new QwenImageParams { NegativePrompt = "no" })).Message);
        // A mask is taken (stage S6); one that is not <Picture 1>'s size is refused before any component is made.
        Assert.Contains("same width and height", Assert.Throws<ArgumentException>(() =>
            runtime.EditImage("x", new[] { picture }, new QwenImageParams { Mask = new RgbImage(32, 32, new float[32 * 32 * 3]) })).Message);
        // The legacy refusals, in the legacy order, before any component is made.
        Assert.Contains("cfg must be 1", Assert.Throws<ArgumentException>(() => runtime.GenerateImage("x", new QwenImageParams { CfgScale = 2 })).Message);
        Assert.Contains("negative prompt", Assert.Throws<ArgumentException>(() => runtime.GenerateImage("x", new QwenImageParams { NegativePrompt = "no" })).Message);
        Assert.Contains("requires a first reference image", Assert.Throws<ArgumentException>(() =>
            runtime.GenerateImage("x", new QwenImageParams { Mask = new RgbImage(1, 1, new float[3]) })).Message);
    }

    /// <summary>A GGUF denoiser (unlabelled, as the published quantized files' tensors are read) builds the same model.</summary>
    [PinnedGgmlServesFact(MiniMaxH3ImageFamily.ArchitectureId)]
    public void AGgufDenoiserLoadsAsTheRuntimesModel()
    {
        using var dit = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.Q8_0);
        using var vae = new MiniMaxH3ImageVaeFixture(2);
        Companions(dit.Dir, vae);
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(dit.DitPath));
        using ModelBase model = ModelBase.Create(dit.DitPath, TestGates.PinnedGgmlBackend);
        Assert.IsType<MiniMaxH3ImageDiffusionModel>(model);
    }

    /// <summary>ggml_vulkan is not in the serving contract: the model, its denoiser and its VAE refuse it at load, naming the
    /// backend by its option spelling. (The tests that run the family on the pinned GGML backend skip in this lane.)</summary>
    [GgmlFact(BackendType.GgmlVulkan)]
    public void GgmlVulkanIsRefusedAtLoad()
    {
        Assert.DoesNotContain(BackendType.GgmlVulkan,
            ModelArchitectureRegistry.All.Single(a => a.Id == MiniMaxH3ImageFamily.ArchitectureId).Serving.Backends);
        using var dit = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.BF16, safetensors: true);
        using var vae = new MiniMaxH3ImageVaeFixture(2);
        Companions(dit.Dir, vae);
        var model = Assert.Throws<ModelLoadRefusedException>(() => ModelBase.Create(dit.DitPath, BackendType.GgmlVulkan));
        Assert.Contains("runs on --backend ggml_cuda, ggml_metal or ggml_cpu, not ggml_vulkan", model.Message);
        var denoiser = Assert.Throws<NotSupportedException>(() => new MiniMaxH3ImageDenoiser(dit.DitPath, BackendType.GgmlVulkan));
        Assert.EndsWith("not ggml_vulkan.", denoiser.Message);
        var codec = Assert.Throws<NotSupportedException>(() => new MiniMaxH3ImageVaeCodec(vae.VaePath, BackendType.GgmlVulkan));
        Assert.EndsWith("not ggml_vulkan.", codec.Message);
    }

    /// <summary>Multi-GPU requests are refused by the contract's gaps, as for any family, before anything loads.</summary>
    [Fact]
    public void TensorParallelAndLayerSplitAreTheContractsGaps()
    {
        using var dit = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.BF16, safetensors: true);
        var tp = Assert.Throws<NotSupportedException>(() => ModelBase.Create(dit.DitPath, BackendType.GgmlCuda, tpDegree: 2));
        Assert.Contains("tensor-parallel", tp.Message);
        var split = Assert.Throws<NotSupportedException>(() => ModelBase.Create(dit.DitPath, BackendType.GgmlCuda, layerSplitDegree: 2));
        Assert.Contains("layer-split", split.Message);
    }

    /// <summary>The family's detector: the H3 DiT's video and text projections under one prefix, and no audio stream.</summary>
    [Fact]
    public void TheFamilysDetectorTellsTheImageDenoiserFromTheVideoOne()
    {
        Assert.True(MiniMaxH3ImageFamily.IsDenoiser(new[] { "video_patch_proj.weight", "condition_proj.weight", "blocks.0.norm1.weight" }));
        Assert.True(MiniMaxH3ImageFamily.IsDenoiser(new[] { "model.diffusion_model.video_patch_proj.weight", "model.diffusion_model.condition_proj.weight" }));
        Assert.False(MiniMaxH3ImageFamily.IsDenoiser(new[] { "video_patch_proj.weight", "condition_proj.weight", "audio_patch_proj.weight" }));
        Assert.False(MiniMaxH3ImageFamily.IsDenoiser(new[] { "video_patch_proj.weight" }));
        Assert.False(MiniMaxH3ImageFamily.IsDenoiser(new[] { "decoder.x_embedder.weight" }));
        using var vae = new MiniMaxH3ImageVaeFixture(1);
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(vae.VaePath));
    }
}
