// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// unsloth/MiniMax-H3-Image-GGUF, made with ComfyUI-GGUF's converter, declares
// general.architecture = hyvid. The registry dispatches a labelled file on its label
// before it looks at tensors, so the file died at load with "Unsupported architecture:
// hyvid" although its tensors are exactly what MiniMaxH3ImageModel reads. These tests
// pin the fix: a hyvid-labelled H3-Image denoiser resolves to minimax-h3-image, while
// hyvid-labelled files with other tensors stay refused.
using System;
using System.IO;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class MiniMaxH3ImageGgufLabelTests : IDisposable
{
    private readonly string _dir;

    public MiniMaxH3ImageGgufLabelTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-h3-image-hyvid-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // The tensors the H3 detectors key on: the image denoiser has the video patch and
    // text projections, the audio-video one the audio patch projection as well.
    private static readonly string[] ImageDenoiser =
        { "video_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight" };

    private static readonly string[] AudioVideoDenoiser =
        { "video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight", "blocks.0.attn.qkv_proj.weight" };

    private string Write(string name, string? architecture, string[] tensors)
        => LoraHostFixtures.WriteGguf(Path.Combine(_dir, name), architecture, tensors);

    private static ModelArchitectureDescriptor Resolve(string path)
    {
        using var probe = new GgufFile(path);
        return ModelArchitectureRegistry.Resolve(probe.GetString("general.architecture"), probe);
    }

    [Fact]
    public void HyvidIsNotAnAlias_ItIsClaimedPerFile()
    {
        Assert.False(ModelArchitectureRegistry.TryGet("hyvid", out _));
    }

    [Fact]
    public void HyvidLabelledImageDenoiser_ResolvesToMiniMaxH3Image()
    {
        string path = Write("minimax_h3_image_pruned-Q8_0.gguf", "hyvid", ImageDenoiser);
        Assert.Equal(MiniMaxH3ImageModel.ArchitectureId, Resolve(path).Id);
    }

    [Fact]
    public void UnlabelledImageDenoiser_StillResolvesByItsTensors()
    {
        string path = Write("unlabelled.gguf", null, ImageDenoiser);
        Assert.Equal(MiniMaxH3ImageModel.ArchitectureId, Resolve(path).Id);
    }

    [Fact]
    public void HyvidLabelledAudioVideoDenoiser_IsNotTakenForTheImageModel()
    {
        string path = Write("hyvid-audio-video.gguf", "hyvid", AudioVideoDenoiser);
        var ex = Assert.Throws<NotSupportedException>(() => Resolve(path));
        Assert.Contains("hyvid", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void HyvidLabelledOtherModel_IsRefusedWithWhatIsAccepted()
    {
        string path = Write("hunyuan-video.gguf", "hyvid", new[] { "img_in.proj.weight", "txt_in.input_embedder.weight" });
        var ex = Assert.Throws<NotSupportedException>(() => Resolve(path));
        Assert.Contains("Unsupported architecture: hyvid", ex.Message, StringComparison.Ordinal);
        // The refusal names what a 'hyvid' file has to be to load.
        Assert.Contains("ComfyUI-GGUF", ex.Message, StringComparison.Ordinal);
    }
}
