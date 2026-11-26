// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron vision mmproj layouts (PR #282 review item 3). Bolt's mmproj stores each block's
// attention projection split into attn_q/attn_k/attn_v weights and biases, where the Nano Omni
// and Super VL mmprojs store one fused attn_qkv; the encoder read only attn_qkv, so every Bolt
// image request died with a NullReferenceException (HTTP 500) and startup printed "Nemotron
// multimodal warmup skipped". Split files are now fused at load and must encode exactly what
// the fused file encodes; a file the encoder cannot run is refused at load, naming the tensors.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Mmproj = InferenceWeb.Tests.NemotronSyntheticMmprojBuilder;

namespace InferenceWeb.Tests;

public sealed class NemotronVisionEncoderTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-vision-" + Guid.NewGuid().ToString("N"));

    public NemotronVisionEncoderTests() => Directory.CreateDirectory(_directory);

    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* best effort */ }
    }

    // The managed rows run in every lane. The native rows construct a GGML backend, and a
    // process holds one, so they run in the lane that pins it and skip visibly elsewhere.
    [Theory]
    [InlineData((int)SyntheticGguf.GgmlType.F32)]
    [InlineData((int)SyntheticGguf.GgmlType.BF16)]
    [InlineData((int)SyntheticGguf.GgmlType.Q8_0)]
    public void SplitQkvMmproj_EncodesBitIdenticallyToFusedMmproj(int qkvType)
        => CheckSplitEncodesLikeFused((SyntheticGguf.GgmlType)qkvType, () => new CpuAllocator(BlasEnum.DotNet));

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData((int)SyntheticGguf.GgmlType.F32)]
    [InlineData((int)SyntheticGguf.GgmlType.Q8_0)]
    public void SplitQkvMmproj_EncodesBitIdenticallyToFusedMmproj_GgmlCpu(int qkvType)
        => CheckSplitEncodesLikeFused((SyntheticGguf.GgmlType)qkvType,
            () => new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cpu), 0));

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData((int)SyntheticGguf.GgmlType.F32)]
    [InlineData((int)SyntheticGguf.GgmlType.BF16)]
    public void SplitQkvMmproj_EncodesBitIdenticallyToFusedMmproj_GgmlCuda(int qkvType)
        => CheckSplitEncodesLikeFused((SyntheticGguf.GgmlType)qkvType,
            () => new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Cuda), 0));

    private void CheckSplitEncodesLikeFused(SyntheticGguf.GgmlType qkvType, Func<IAllocator> allocator)
    {
        string fusedPath = new Mmproj { QkvType = qkvType }.Write(Path.Combine(_directory, "fused.gguf"));
        // Bolt-like: split q/k/v and the video tower's patch embedding, which is skipped.
        string splitPath = new Mmproj { QkvType = qkvType, SplitQkv = true, VideoPatchEmbedding = true }
            .Write(Path.Combine(_directory, "split.gguf"));
        float[] pixels = Tile();

        IAllocator shared = allocator();
        using var fused = new NemotronVisionEncoder(fusedPath, shared);
        using var split = new NemotronVisionEncoder(splitPath, shared);
        using var expected = fused.Encode(pixels, Mmproj.ImageSize, Mmproj.ImageSize);
        using var actual = split.Encode(pixels, Mmproj.ImageSize, Mmproj.ImageSize);

        Assert.Equal(new long[] { Mmproj.TokensPerTile, NemotronSyntheticModelBuilder.Hidden }, expected.Sizes.ToArray());
        Assert.Equal(expected.Sizes.ToArray(), actual.Sizes.ToArray());
        float[] e = expected.GetElementsAsFloat((int)expected.ElementCount());
        float[] a = actual.GetElementsAsFloat((int)actual.ElementCount());
        Assert.All(e, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(e, v => v != 0f);
        for (int i = 0; i < e.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(e[i]) == BitConverter.SingleToInt32Bits(a[i]),
                $"{qkvType}: element {i} split={a[i]:R} fused={e[i]:R}");
    }

    [Fact]
    public void SplitQkvMmproj_StoresOneFusedProjectionPerBlock_AndSkipsTheVideoPatchEmbedding()
    {
        string path = new Mmproj { SplitQkv = true, VideoPatchEmbedding = true }.Write(Path.Combine(_directory, "split.gguf"));
        using var encoder = new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet));
        var weights = Weights(encoder);

        for (int b = 0; b < Mmproj.Blocks; b++)
        {
            Assert.Equal(new long[] { 3 * Mmproj.Hidden, Mmproj.Hidden }, weights[$"v.blk.{b}.attn_qkv.weight"].Sizes.ToArray());
            Assert.Equal(new long[] { 3 * Mmproj.Hidden }, weights[$"v.blk.{b}.attn_qkv.bias"].Sizes.ToArray());
            foreach (string module in new[] { "attn_q", "attn_k", "attn_v" })
            {
                Assert.False(weights.ContainsKey($"v.blk.{b}.{module}.weight"));
                Assert.False(weights.ContainsKey($"v.blk.{b}.{module}.bias"));
            }
        }
        Assert.False(weights.ContainsKey("v.patch_embd.weight.1"));
        Assert.Equal(NemotronSyntheticModelBuilder.Hidden, encoder.ProjectionDim);
    }

    [Fact]
    public void SuperVlLikeMmproj_StillLoadsAndEncodes()
    {
        // Super VL's ggml-org mmproj: fused q/k/v plus the video patch embedding (and
        // v.post_ln, which this encoder does not read). Validation must not refuse it.
        string path = new Mmproj { VideoPatchEmbedding = true }.Write(Path.Combine(_directory, "fused-video.gguf"));
        using var encoder = new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet));
        using var embeddings = encoder.Encode(Tile(), Mmproj.ImageSize, Mmproj.ImageSize);
        Assert.Equal(new long[] { Mmproj.TokensPerTile, NemotronSyntheticModelBuilder.Hidden }, embeddings.Sizes.ToArray());
        Assert.All(embeddings.GetElementsAsFloat((int)embeddings.ElementCount()), v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void SplitQkvNames_MapToTheFusedProjectionInQkvOrder()
    {
        Assert.True(NemotronVisionEncoder.TryParseSplitQkvName("v.blk.0.attn_q.weight", out string fused, out int part));
        Assert.Equal(("v.blk.0.attn_qkv.weight", 0), (fused, part));
        Assert.True(NemotronVisionEncoder.TryParseSplitQkvName("v.blk.7.attn_k.weight", out fused, out part));
        Assert.Equal(("v.blk.7.attn_qkv.weight", 1), (fused, part));
        Assert.True(NemotronVisionEncoder.TryParseSplitQkvName("v.blk.31.attn_v.bias", out fused, out part));
        Assert.Equal(("v.blk.31.attn_qkv.bias", 2), (fused, part));

        foreach (string other in new[]
        {
            "v.blk.0.attn_qkv.weight", "v.blk.0.attn_out.weight", "v.blk.0.attn_q.scale", "v.blk.x.attn_q.weight",
            "v.blk..attn_q.weight", "v.blk.0.attn_q", "blk.0.attn_q.weight", "mm.model.mlp.0.weight", "v.patch_embd.weight.1",
        })
        {
            Assert.False(NemotronVisionEncoder.TryParseSplitQkvName(other, out _, out _), other);
        }

        Assert.Equal(new[] { 1f, 2f, 3f, 4f, 5f, 6f }, NemotronVisionEncoder.ConcatRows(new[] { 1f, 2f }, new[] { 3f, 4f }, new[] { 5f, 6f }));
    }

    [Theory]
    [InlineData("v.blk.1.attn_v.weight")]
    [InlineData("v.blk.0.attn_k.bias")]
    public void IncompleteSplitTriple_IsRefusedAtLoad_NamingTheMissingPart(string omitted)
    {
        string path = new Mmproj { SplitQkv = true, Omit = new[] { omitted } }.Write(Path.Combine(_directory, "partial.gguf"));
        var error = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet)));
        Assert.Contains("partial.gguf", error.Message);
        Assert.Contains("without " + omitted, error.Message);
        Assert.Contains("complete split attn_q/attn_k/attn_v triple", error.Message);
    }

    [Fact]
    public void FusedAndSplitProjectionInOneBlock_IsRefusedAtLoad()
    {
        string path = new Mmproj { FusedAndSplitBlock0 = true }.Write(Path.Combine(_directory, "both.gguf"));
        var error = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet)));
        Assert.Contains("both.gguf", error.Message);
        Assert.Contains("v.blk.0.attn_qkv.weight and split v.blk.0.attn_q.weight/v.blk.0.attn_k.weight/v.blk.0.attn_v.weight are both present", error.Message);
        Assert.Contains("v.blk.0.attn_qkv.bias and split", error.Message);
        Assert.DoesNotContain("v.blk.1.", error.Message);
    }

    [Fact]
    public void MissingRequiredTensors_AreRefusedAtLoad_WithTheFileAndExpectedLayout()
    {
        string path = new Mmproj
        {
            SplitQkv = true,
            Omit = new[] { "v.blk.1.ffn_down.weight", "mm.model.mlp.0.weight", "v.blk.0.ln2.weight", "v.blk.0.ffn_up.bias" },
        }.Write(Path.Combine(_directory, "missing.gguf"));
        var error = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet)));
        Assert.Contains("missing.gguf", error.Message);
        Assert.Contains("Missing 3 required tensor(s): v.blk.0.ln2.weight, v.blk.1.ffn_down.weight, mm.model.mlp.0.weight.", error.Message);
        // Biases stay optional.
        Assert.DoesNotContain("ffn_up.bias", error.Message);
        Assert.Contains("Expected layout (GGUF ne order)", error.Message);
        Assert.Contains($"attn_qkv.weight [{Mmproj.Hidden}, {3 * Mmproj.Hidden}]", error.Message);

        // A file missing whole tensor families lists twelve and counts the rest.
        var everyBlockTensor = Enumerable.Range(0, Mmproj.Blocks)
            .SelectMany(b => new[] { "ln1", "attn_out", "ln2", "ffn_up", "ffn_down" }.Select(m => $"v.blk.{b}.{m}.weight"))
            .Concat(new[] { "mm.model.mlp.0.weight", "mm.model.mlp.1.weight", "mm.model.mlp.3.weight", "v.patch_embd.weight" })
            .ToArray();
        string gutted = new Mmproj { Omit = everyBlockTensor }.Write(Path.Combine(_directory, "gutted.gguf"));
        error = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(gutted, new CpuAllocator(BlasEnum.DotNet)));
        Assert.Contains($"Missing {everyBlockTensor.Length} required tensor(s)", error.Message);
        Assert.Contains($"(and {everyBlockTensor.Length - 12} more)", error.Message);
    }

    [Fact]
    public void TensorsDisagreeingWithTheMetadata_AreRefusedAtLoad()
    {
        // The file declares a 48-wide feed-forward; its tensors are 64 wide.
        string path = new Mmproj { DeclaredFeedForward = 48 }.Write(Path.Combine(_directory, "ffn.gguf"));
        var error = Assert.Throws<InvalidDataException>(() => new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet)));
        Assert.Contains("ffn.gguf", error.Message);
        Assert.Contains($"v.blk.0.ffn_up.weight is [{Mmproj.Hidden}, {Mmproj.Intermediate}], expected [{Mmproj.Hidden}, 48]", error.Message);
        Assert.Contains("v.blk.0.ffn_up.bias is [64], expected [48]", error.Message);
    }

    [Fact]
    public void EncodeWithoutAWeight_ThrowsNamingIt_InsteadOfNullReference()
    {
        // Defence in depth behind the load-time check: the old LinearForward returned null
        // and SelfAttention dereferenced it.
        string path = new Mmproj().Write(Path.Combine(_directory, "fused.gguf"));
        using var encoder = new NemotronVisionEncoder(path, new CpuAllocator(BlasEnum.DotNet));
        var weights = Weights(encoder);
        weights["v.blk.1.attn_out.weight"].Dispose();
        weights.Remove("v.blk.1.attn_out.weight");

        var error = Assert.Throws<InvalidOperationException>(() => encoder.Encode(Tile(), Mmproj.ImageSize, Mmproj.ImageSize));
        Assert.Contains("v.blk.1.attn_out.weight", error.Message);
    }

    [Fact]
    public void SplitAndFusedMmproj_GiveTheLanguageModelTheSameImagePrefill()
    {
        string lm = new NemotronSyntheticModelBuilder().Write(Path.Combine(_directory, "lm.gguf"));
        string fusedPath = new Mmproj().Write(Path.Combine(_directory, "fused.gguf"));
        string splitPath = new Mmproj { SplitQkv = true, VideoPatchEmbedding = true }.Write(Path.Combine(_directory, "split.gguf"));

        float[] Prefill(string mmproj)
        {
            using var model = new NemotronRuntimeModel(lm, BackendType.Cpu);
            // LoadVisionEncoder directly: LoadProjectors would also consult the
            // process-wide TS_NEMOTRON_AUDIO_MMPROJ.
            model.LoadVisionEncoder(mmproj);
            Assert.NotNull(model.VisionEncoder);
            // <img> + one tile's image tokens + </img>; Forward takes ownership of the rows.
            var tile = model.VisionEncoder.Encode(Tile(), Mmproj.ImageSize, Mmproj.ImageSize);
            model.SetVisionEmbeddings(tile, 1);
            var tokens = new int[Mmproj.TokensPerTile + 2];
            Array.Fill(tokens, 40);
            tokens[0] = 41;
            tokens[^1] = 42;
            float[] logits = model.Forward(tokens);
            Assert.All(logits, v => Assert.True(float.IsFinite(v)));
            return logits;
        }

        float[] fused = Prefill(fusedPath);
        float[] split = Prefill(splitPath);
        Assert.Equal(fused.Length, split.Length);
        for (int i = 0; i < fused.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(fused[i]) == BitConverter.SingleToInt32Bits(split[i]),
                $"logit {i}: split={split[i]:R} fused={fused[i]:R}");
    }

    [Fact]
    public void MmprojOfAnotherProjectionWidth_IsRefusedAtLoad()
    {
        // e.g. a Nano Omni mmproj (2688) beside a Bolt or Super VL language model (4096).
        string lm = new NemotronSyntheticModelBuilder().Write(Path.Combine(_directory, "lm.gguf"));
        string mmproj = new Mmproj { ProjectionDim = 32 }.Write(Path.Combine(_directory, "narrow.gguf"));
        using var model = new NemotronRuntimeModel(lm, BackendType.Cpu);

        var error = Assert.Throws<InvalidDataException>(() => model.LoadVisionEncoder(mmproj));
        Assert.Contains("narrow.gguf", error.Message);
        Assert.Contains("32-wide embeddings", error.Message);
        Assert.Contains($"hidden size is {NemotronSyntheticModelBuilder.Hidden}", error.Message);
        Assert.Null(model.VisionEncoder);
    }

    /// <summary>A deterministic normalized [C, H, W] tile.</summary>
    private static float[] Tile()
    {
        var pixels = new float[Mmproj.Channels * Mmproj.ImageSize * Mmproj.ImageSize];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = MathF.Sin(i * 0.37f) * 1.5f;
        return pixels;
    }

    private static Dictionary<string, Tensor> Weights(NemotronVisionEncoder encoder)
        => (Dictionary<string, Tensor>)typeof(NemotronVisionEncoder)
            .GetField("_weights", BindingFlags.NonPublic | BindingFlags.Instance)!
            .GetValue(encoder)!;
}
