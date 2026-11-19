// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Half = System.Half;

namespace InferenceWeb.Tests;

/// <summary>
/// A tiny MiniMax-H3-Image VAE decoder written at test time from a seed (docs/design/diffusion-runtime.md section 7.4): the
/// release file's tensor names, shapes in torch order and its all-F16 storage (Comfy-Org vae/minimax_h3_image_vae_fp16), with
/// the release's head size (64, RoPE on 32 dims) and 16x16 RGB patches, but width 128 (2 heads), FFN 256 and
/// <see cref="Blocks"/> blocks.
/// </summary>
internal sealed class MiniMaxH3ImageVaeFixture : IDisposable
{
    internal const int Dim = 128, Inner = 256, Latent = 24, Registers = 4, Patch = 768;

    internal MiniMaxH3ImageVaeFixture(int blocks, int seed = 3071)
    {
        Blocks = blocks;
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3vae-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
        VaePath = Path.Combine(Dir, "minimax_h3_image_vae_tiny.safetensors");
        Write(VaePath, Tensors(blocks, seed));
    }

    internal string Dir { get; }
    internal int Blocks { get; }
    internal string VaePath { get; }

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>MiniMaxH3ImageVae's native graph budget for <paramref name="batch"/> tiles (h3_vit_decode: room for the fused
    /// cuDNN node only); a decode whose unfused graph needs more aborts the process off ggml_cuda's fused path.</summary>
    internal static long LegacyGraphBudget(int blocks, int batch) => (long)blocks * (104 + 8 * (batch - 1)) + 2048 + 8 * (batch - 1);

    private static List<(string Name, long[] Shape, float[] Data)> Tensors(int blocks, int seed)
    {
        var rng = new Random(seed);
        var tensors = new List<(string, long[], float[])>();
        float[] Values(long n, float scale, float offset) =>
            Enumerable.Range(0, checked((int)n)).Select(_ => offset + scale * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        void Add(string name, float scale, float offset, params long[] shape) =>
            tensors.Add((name, shape, Values(shape.Aggregate(1L, (a, b) => a * b), scale, offset)));
        // A projection [out, in] (torch order) with values about 1 / sqrt(in).
        void Proj(string name, long output, long input) => Add(name, 1.2f / MathF.Sqrt(input), 0f, output, input);

        Proj("post_quant_conv.weight", Latent, Latent);
        tensors[^1] = (tensors[^1].Item1, new long[] { Latent, Latent, 1, 1 }, tensors[^1].Item3);
        Add("post_quant_conv.bias", 0.05f, 0f, Latent);
        Proj("decoder.x_embedder.weight", Dim, Latent);
        Add("decoder.x_embedder.bias", 0.05f, 0f, Dim);
        Add("decoder.register_tokens", 0.5f, 0f, 1, Registers, Dim);
        Add("decoder.mask_token", 0.5f, 0f, 1, 1, Dim);   // in the release, unused by the decoder
        for (int b = 0; b < blocks; b++)
        {
            string p = $"decoder.transformer_blocks.{b}.";
            Add(p + "norm1.weight", 0.2f, 1f, Dim);
            Add(p + "norm2.weight", 0.2f, 1f, Dim);
            Add(p + "scale1", 0.2f, 0.3f, Dim);
            Add(p + "scale2", 0.2f, 0.3f, Dim);
            Proj(p + "attn.to_qkv.weight", 3 * Dim, Dim);
            Add(p + "attn.to_qkv.bias", 0.1f, 0f, 3 * Dim);
            Proj(p + "attn.to_out.weight", Dim, Dim);
            Add(p + "attn.to_out.bias", 0.05f, 0f, Dim);
            Proj(p + "ff.w1.weight", 2 * Inner, Dim);
            Add(p + "ff.w1.bias", 0.05f, 0f, 2 * Inner);
            Proj(p + "ff.w2.weight", Dim, Inner);
            Add(p + "ff.w2.bias", 0.05f, 0f, Dim);
        }
        Add("decoder.norm_out.weight", 0.2f, 1f, Dim);
        Add("decoder.norm_out.bias", 0.05f, 0f, Dim);
        Proj("decoder.proj_out.weight", Patch, Dim);
        Add("decoder.proj_out.bias", 0.05f, 0f, Patch);
        return tensors;
    }

    /// <summary>A safetensors file of F16 tensors in order.</summary>
    private static void Write(string path, List<(string Name, long[] Shape, float[] Data)> tensors)
    {
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, shape, data) in tensors)
        {
            long bytes = data.LongLength * 2;
            header[name] = new Dictionary<string, object> { ["dtype"] = "F16", ["shape"] = shape, ["data_offsets"] = new[] { offset, offset + bytes } };
            offset += bytes;
        }
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using var file = File.Create(path);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)json.Length);
        file.Write(length);
        file.Write(json);
        foreach (var (_, _, data) in tensors)
            file.Write(MemoryMarshal.AsBytes(data.Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray().AsSpan()));
    }
}
