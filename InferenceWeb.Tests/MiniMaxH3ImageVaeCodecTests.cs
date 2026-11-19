// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image's VAE decoder on the diffusion runtime (MiniMaxH3ImageVaeCodec: a row of tiles a GraphProgram) against
/// the legacy decoder (MiniMaxH3ImageVae, h3_vit_decode's native graph) on a tiny synthetic decoder
/// (<see cref="MiniMaxH3ImageVaeFixture"/>): the same pixels bit for bit on every GGML backend, with ComfyUI's tiling and
/// blending, at the batch sizes the legacy graph cannot hold off ggml_cuda's fused path (it is compared there with the
/// legacy decoder four tiles a call, which decodes the same tiles).
/// </summary>
public sealed class MiniMaxH3ImageVaeCodecTests(ITestOutputHelper output)
{
    /// <summary>Enough blocks that a row of 11 tiles overflows the legacy graph's node budget off the fused path.</summary>
    private const int Blocks = 8;

    private static float[] NormalizedLatent(int h, int w, int seed)
    {
        var rng = new Random(seed);
        var v = new float[(long)h * w * MiniMaxH3ImageVaeCodec.LatentChannels];
        for (long i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static void Pin(BackendType backend) => GgmlBasicOps.EnsureBackendAvailable(backend switch
    {
        BackendType.GgmlMetal => GgmlBackendType.Metal,
        BackendType.GgmlCuda => GgmlBackendType.Cuda,
        _ => GgmlBackendType.Cpu,
    });

    private void AssertSamePixels(RgbImage want, RgbImage got, string what)
    {
        Assert.Equal(want.Width, got.Width);
        Assert.Equal(want.Height, got.Height);
        int differ = 0, first = -1;
        for (int i = 0; i < want.Pixels.Length; i++)
            if (BitConverter.SingleToInt32Bits(want.Pixels[i]) != BitConverter.SingleToInt32Bits(got.Pixels[i]))
            {
                if (first < 0) first = i;
                differ++;
            }
        double mean = 0;
        foreach (float v in want.Pixels) mean += v;
        output.WriteLine($"{what}: {got.Width}x{got.Height}, {differ} of {want.Pixels.Length} values differ (mean pixel {mean / want.Pixels.Length:F4})");
        Assert.True(differ == 0, $"{what}: {differ} values differ, the first at {first} (legacy {(first >= 0 ? want.Pixels[first] : 0)}, runtime {(first >= 0 ? got.Pixels[first] : 0)})");
    }

    /// <summary>One canvas through both decoders on <paramref name="backend"/>: the legacy one <paramref name="legacyBatch"/>
    /// tiles a call (0: its default, a row), the runtime one a whole row a program.</summary>
    private MiniMaxH3ImageVaeCodec VerifyBitExact(BackendType backend, MiniMaxH3ImageVaeFixture fx, int latentH, int latentW, int legacyBatch,
        int seed = 11)
    {
        Pin(backend);
        float[] latent = NormalizedLatent(latentH, latentW, seed);
        RgbImage want;
        using (var legacy = new MiniMaxH3ImageVae(fx.VaePath) { BatchOverride = legacyBatch })
            want = legacy.Decode(latent, latentH, latentW);
        var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend);
        try
        {
            RgbImage got = codec.Decode(latent, latentH, latentW);
            AssertSamePixels(want, got, $"{backend} {latentW * 16}x{latentH * 16} (legacy {(legacyBatch > 0 ? legacyBatch.ToString() : "a row")} a call)");
            // The canvas's rows share a tile shape and a batch: one program, replayed row after row.
            Assert.Equal(1, codec.Compiles);
            return codec;
        }
        catch
        {
            codec.Dispose();
            throw;
        }
    }

    /// <summary>A row of 11 tiles (a 2048-pixel-wide canvas) is one program on ggml_cpu, whose graph the legacy decoder could
    /// not build at that batch (its budget assumes the fused cuDNN node: GGML_ASSERT n_nodes &lt; size), and its pixels are
    /// the legacy decoder's four tiles at a time.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void ARowOfElevenTilesDecodesAsOneProgramOnGgmlCpu() => ARowOfElevenTiles(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void ARowOfElevenTilesDecodesAsOneProgramOnGgmlMetal() => ARowOfElevenTiles(BackendType.GgmlMetal);

    private void ARowOfElevenTiles(BackendType backend)
    {
        using var fx = new MiniMaxH3ImageVaeFixture(Blocks);
        using MiniMaxH3ImageVaeCodec codec = VerifyBitExact(backend, fx, 16, 128, legacyBatch: 4);
        Assert.Equal(11, codec.LastTileCount);
        long budget = MiniMaxH3ImageVaeFixture.LegacyGraphBudget(Blocks, 11);
        output.WriteLine($"program nodes {codec.LastProgramNodes}, the legacy graph's budget at 11 tiles {budget}");
        Assert.True(codec.LastProgramNodes > budget, "the fixture must be one the legacy graph cannot hold at a row's batch");
        Assert.Equal(0, GgmlGraphExecutor.ServedRegions(codec.LastPlan));
        // Each tile's attention of each block is the native decoder's (h3_attend).
        Assert.Equal(11 * Blocks, GgmlGraphExecutor.LoweredRegions(codec.LastPlan));
    }

    /// <summary>On ggml_cuda the regions are the legacy decoder's fused nodes (the F16 products on cuBLAS, one cuDNN attention
    /// over the row, the scaled residuals), and both decode a row a call.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void ARowOfElevenTilesIsTheLegacyFusedDecodeOnGgmlCuda()
    {
        using var fx = new MiniMaxH3ImageVaeFixture(Blocks);
        using MiniMaxH3ImageVaeCodec codec = VerifyBitExact(BackendType.GgmlCuda, fx, 16, 128, legacyBatch: 0);
        int served = GgmlGraphExecutor.ServedRegions(codec.LastPlan);
        output.WriteLine($"{served} regions served, {GgmlGraphExecutor.LoweredRegions(codec.LastPlan)} lowered");
        Assert.Equal(3 * Blocks, served);   // per block: the attention and the two scaled residuals
    }

    public static IEnumerable<object[]> Canvases() => new[]
    {
        new object[] { 32, 40 },   // 640 x 512: rows of three tiles with uneven overlaps, blended both ways
        new object[] { 16, 12 },   // 192 x 256: one tile, the decoder's single-sequence graph
        new object[] { 20, 36 },   // 576 x 320: two rows of three
    };

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Canvases))]
    public void CanvasesDecodeAsTheLegacyDecoderOnGgmlCpu(int latentH, int latentW)
    {
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        using MiniMaxH3ImageVaeCodec _ = VerifyBitExact(BackendType.GgmlCpu, fx, latentH, latentW, legacyBatch: 0);
    }

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Canvases))]
    public void CanvasesDecodeAsTheLegacyDecoderOnGgmlMetal(int latentH, int latentW)
    {
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        using MiniMaxH3ImageVaeCodec _ = VerifyBitExact(BackendType.GgmlMetal, fx, latentH, latentW, legacyBatch: 0);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Canvases))]
    public void CanvasesDecodeAsTheLegacyDecoderOnGgmlCuda(int latentH, int latentW)
    {
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        using MiniMaxH3ImageVaeCodec _ = VerifyBitExact(BackendType.GgmlCuda, fx, latentH, latentW, legacyBatch: 0);
    }

    /// <summary>A row longer than the batch limit takes two programs (the limit and the rest), both cached, and decodes as
    /// the legacy decoder does with the same limit. (Off ggml_cuda a row's pixels do not depend on its split; on ggml_cuda
    /// cuDNN's batched attention plans by batch, so they do, for both decoders alike.)</summary>
    [Fact]
    public void ARowSplitAcrossProgramsDecodesAsTheLegacyDecoderSplitTheSameWay()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Pin(backend);
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        float[] latent = NormalizedLatent(16, 128, 5);
        RgbImage want;
        using (var legacy = new MiniMaxH3ImageVae(fx.VaePath) { BatchOverride = 4 })
            want = legacy.Decode(latent, 16, 128);
        using var split = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend) { BatchLimit = 4 };
        RgbImage got = split.Decode(latent, 16, 128);
        AssertSamePixels(want, got, $"{backend} 11 tiles as 4 + 4 + 3");
        Assert.Equal(2, split.Compiles);
        split.Decode(latent, 16, 128);
        Assert.Equal(2, split.Compiles);
        if (backend == BackendType.GgmlCuda) return;
        using var whole = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend);
        AssertSamePixels(want, whole.Decode(latent, 16, 128), $"{backend} 11 tiles in one program");
    }

    /// <summary>Released workspaces and device copies are made again by the next decode, to the same pixels; the latent's
    /// [C, 1, H, W] form decodes as its channel-last one.</summary>
    [Fact]
    public void ReleasesAndTheLatentTensorFormDecodeTheSamePixels()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Pin(backend);
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        float[] hwc = NormalizedLatent(20, 20, 9);
        using var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend);
        RgbImage want = codec.Decode(hwc, 20, 20);
        codec.ReleaseWorkspace();
        Assert.Null(codec.LastPlan);
        AssertSamePixels(want, codec.Decode(hwc, 20, 20), "after a released workspace");
        codec.ReleaseDevice();
        var chw = new float[hwc.Length];
        for (int c = 0; c < 24; c++)
            for (int i = 0; i < 400; i++)
                chw[(long)c * 400 + i] = hwc[(long)i * 24 + c];
        AssertSamePixels(want, codec.Decode(new LatentTensor(new LatentShape(24, 1, 20, 20), chw), CancellationToken.None),
            "the [C, 1, H, W] latent after released device copies");
        Assert.Equal(3, codec.Compiles);
        RgbImage preview = codec.Preview(new LatentTensor(new LatentShape(24, 1, 20, 20), chw), 160);
        Assert.Equal((160, 160), (preview.Width, preview.Height));
    }

    [Fact]
    public void TheCodecDeclaresItsWeightsAndARowsWorkingMemoryAndRefusesToEncode()
    {
        using var fx = new MiniMaxH3ImageVaeFixture(2);
        using var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, TestGates.PinnedGgmlBackend);
        Assert.Equal(2, codec.NumBlocks);
        Assert.Equal(2, codec.Heads);
        Assert.Equal(MiniMaxH3ImageVaeFixture.Inner, codec.FfnInner);
        // F16 projections as stored, the vectors widened to F32.
        long projections = 2L * (24 * 24 + 128 * 24 + 768 * 128 + 2 * (3 * 128 * 128 + 128 * 128 + 2 * 256 * 128 + 128 * 256));
        long vectors = 4L * (24 + 128 + 4 * 128 + 128 + 128 + 768 + 2 * (4 * 128 + 3 * 128 + 128 + 2 * 256 + 128));
        Assert.Equal(projections + vectors, codec.WeightBytes);
        var narrow = new DiffusionWorkload(new LatentShape(24, 1, 16, 16), Array.Empty<LatentShape>(), "");
        var wide = new DiffusionWorkload(new LatentShape(24, 1, 16, 128), Array.Empty<LatentShape>(), "");
        Assert.True(codec.WorkingBytes(wide) > codec.WorkingBytes(narrow));
        Assert.Throws<NotSupportedException>(() => codec.Encode(new RgbImage(16, 16, new float[16 * 16 * 3]), CancellationToken.None));
        Assert.Throws<NotSupportedException>(() => new MiniMaxH3ImageVaeCodec(fx.VaePath, BackendType.Cpu));
    }
}
