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

    /// <param name="encoder">Also write the synthetic encoder of eng/validation/MiniMaxH3ImageEdit/vae_encoder_oracle.py
    /// (<see cref="EncoderTensors"/>), whose expected moments are Fixtures/MiniMaxH3Image/vae-encoder-oracle.json.</param>
    internal MiniMaxH3ImageVaeFixture(int blocks, int seed = 3071, bool encoder = false)
    {
        Blocks = blocks;
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3vae-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
        VaePath = Path.Combine(Dir, "minimax_h3_image_vae_tiny.safetensors");
        var tensors = Tensors(blocks, seed);
        if (encoder) tensors.AddRange(EncoderTensors());
        Write(VaePath, tensors);
    }

    /// <summary>The synthetic encoder's channels per level (each one resnet block and a downsample).</summary>
    internal static readonly int[] EncoderChannels = { 32, 64, 64, 64 };

    /// <summary>
    /// The synthetic encoder in the release file's names and torch shapes (3x3 kernels with three temporal taps, of which
    /// only the last reaches a single frame): its values are vae_encoder_oracle.py's hash of the tensor's name and the
    /// element index, so the NumPy oracle has the same weights without a shared generator.
    /// </summary>
    internal static List<(string Name, long[] Shape, float[] Data)> EncoderTensors()
    {
        var tensors = new List<(string, long[], float[])>();
        void Conv(string name, int oc, int ic, int k)
        {
            long[] shape = k == 3 ? new long[] { oc, ic, 3, k, k } : new long[] { oc, ic, 1, k, k };
            long n = shape.Aggregate(1L, (a, b) => a * b);
            tensors.Add((name + ".weight", shape, OracleValues(name + ".weight", n, 1.0 / Math.Sqrt(ic * k * k), 0)));
            tensors.Add((name + ".bias", new long[] { oc }, OracleValues(name + ".bias", oc, 0.05, 0)));
        }
        void Norm(string name, int c)
        {
            tensors.Add((name + ".weight", new long[] { c }, OracleValues(name + ".weight", c, 0.2, 1)));
            tensors.Add((name + ".bias", new long[] { c }, OracleValues(name + ".bias", c, 0.1, 0)));
        }
        Conv("encoder.conv_in", EncoderChannels[0], 3, 3);
        int cin = EncoderChannels[0];
        for (int level = 0; level < EncoderChannels.Length; level++)
        {
            int c = EncoderChannels[level];
            string p = $"encoder.down.{level}.";
            Norm(p + "block.0.norm1", cin);
            Conv(p + "block.0.conv1", c, cin, 3);
            Norm(p + "block.0.norm2", c);
            Conv(p + "block.0.conv2", c, c, 3);
            if (cin != c) Conv(p + "block.0.nin_shortcut", c, cin, 1);
            Conv(p + "downsample.conv", c, c, 3);
            cin = c;
        }
        Norm("encoder.norm_out", cin);
        Conv("encoder.conv_out", 2 * Latent, cin, 3);
        Conv("quant_conv", 2 * Latent, 2 * Latent, 1);
        return tensors;
    }

    /// <summary>vae_encoder_oracle.py's values(): element i of <paramref name="name"/> a hash of (crc32(name), i) mapped to
    /// [-1, 1), times <paramref name="scale"/>, plus <paramref name="offset"/>, as float32 (stored as float16).</summary>
    internal static float[] OracleValues(string name, long count, double scale, double offset)
    {
        uint salt = Crc32(Encoding.ASCII.GetBytes(name));
        var v = new float[count];
        for (long i = 0; i < count; i++)
        {
            ulong h = ((ulong)i * 2654435761UL + salt * 40503UL + 12345UL) & 0xFFFFFFFFUL;
            h ^= h >> 15;
            h = (h * 2246822519UL) & 0xFFFFFFFFUL;
            double u = (h >> 8) / (double)(1 << 24);
            v[i] = (float)(offset + scale * (2.0 * u - 1.0));
        }
        return v;
    }

    /// <summary>zlib.crc32 (CRC-32/ISO-HDLC).</summary>
    private static uint Crc32(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data)
        {
            crc ^= b;
            for (int k = 0; k < 8; k++) crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }
        return ~crc;
    }

    /// <summary>vae_encoder_oracle.py's image(): the synthetic picture's pixels (bytes / 255) [H][W][3].</summary>
    internal static TensorSharp.Models.QwenImage.RgbImage OracleImage(int width, int height)
    {
        var px = new float[(long)width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                    px[((long)y * width + x) * 3 + c] = (byte)((x * 7 + y * 13 + c * 101 + (x * y) % 17) % 256) / 255f;
        return new TensorSharp.Models.QwenImage.RgbImage(width, height, px);
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
