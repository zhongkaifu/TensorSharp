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
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.GGML;
using Half = System.Half;

namespace InferenceWeb.Tests;

/// <summary>
/// A tiny MiniMax-H3-Image denoiser written at test time from a seed (docs/design/diffusion-runtime.md section 7.4): the
/// release's tensor names and dtype mix (BF16 norms and biases, F16 AdaLN projections, F32 patch projection, heads and
/// tables, the big projections in <see cref="Projections"/>' format), 2 blocks, 1 token-refiner block, hidden 256, 4 heads of
/// 64 (RoPE on 48 dims: 3 axes of 8 frequencies), FFN 512 and the 6 PDD heads of the release grid. No metadata, like the
/// published GGUFs.
/// </summary>
internal sealed class MiniMaxH3ImageFixture : IDisposable
{
    internal const int Hidden = 256, Heads = 4, HeadDim = 64, Inner = Heads * HeadDim, Ffn = 512;
    internal const int TextDim = 64, TimeDim = 8, CurveGrid = 17, Channels = 24, PatchDim = 96, PddSteps = 6, InvFreq = 8;
    internal const int Blocks = 2, RefinerBlocks = 1;

    /// <summary>ggml type ids of the formats a fixture's projections take.</summary>
    internal const int F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, BF16 = 30;

    /// <param name="safetensors">Write the release's form instead (a .safetensors file in the original names, torch-order
    /// shapes; F32, F16 or BF16 projections only), as Comfy-Org ships the bf16 denoiser.</param>
    internal MiniMaxH3ImageFixture(int projections, int seed = 2051, bool safetensors = false)
    {
        Projections = projections;
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3img-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
        if (safetensors)
        {
            if (projections is not (F32 or F16 or BF16)) throw new ArgumentException("safetensors holds F32, F16 or BF16 tensors");
            DitPath = System.IO.Path.Combine(Dir, $"minimax_h3_image_tiny-{Name(projections)}.safetensors");
            WriteSafetensors(DitPath, Tensors(projections, seed));
        }
        else
        {
            DitPath = System.IO.Path.Combine(Dir, $"minimax_h3_image_tiny-{Name(projections)}.gguf");
            Write(DitPath, Tensors(projections, seed));
        }
    }

    internal string Dir { get; }
    internal int Projections { get; }

    /// <summary>The fixture file.</summary>
    internal string DitPath { get; }

    internal static string Name(int type) => type switch { F32 => "F32", F16 => "F16", Q8_0 => "Q8_0", Q4_K => "Q4_K", BF16 => "BF16", _ => type.ToString() };

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
    }

    private sealed record T(string Name, long[] Shape, int Type, float[] Data);

    private static List<T> Tensors(int projections, int seed)
    {
        var rng = new Random(seed);
        var tensors = new List<T>();
        float[] Values(long n, float scale, float offset = 0f) =>
            Enumerable.Range(0, checked((int)n)).Select(_ => offset + scale * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        void Add(string name, int type, float scale, float offset, params long[] shape) =>
            tensors.Add(new T(name, shape, type, Values(shape.Aggregate(1L, (a, b) => a * b), scale, offset)));
        // A projection [in, out] with values about 1 / sqrt(in), so activations stay of order one through the blocks.
        void Proj(string name, int type, long input, long output) => Add(name, type, 1.2f / MathF.Sqrt(input), 0f, input, output);
        void Block(string p, bool modulated)
        {
            Add(p + "norm1.weight", BF16, 0.2f, 1f, Hidden);
            Add(p + "norm2.weight", BF16, 0.2f, 1f, Hidden);
            Add(p + "attn.q_norm.weight", BF16, 0.2f, 1f, HeadDim);
            Add(p + "attn.k_norm.weight", BF16, 0.2f, 1f, HeadDim);
            Proj(p + "attn.qkv_proj.weight", projections, Hidden, 3 * Inner);
            Proj(p + "attn.out_proj.weight", projections, Inner, Hidden);
            Proj(p + "mlp.fc1.weight", projections, Hidden, 2 * Ffn);
            Proj(p + "mlp.fc2.weight", projections, Ffn, Hidden);
            if (modulated)
            {
                Add(p + "adaln_proj.linear.weight", F16, 0.3f, 0f, TimeDim, 18 * Hidden);
                Add(p + "adaln_proj.linear.bias", BF16, 0.1f, 0f, 18 * Hidden);
            }
        }
        Add("adaln_t_table", F32, 1f, 0f, TimeDim, CurveGrid);
        Proj("condition_proj.weight", BF16, TextDim, Hidden);
        Add("condition_proj.bias", BF16, 0.05f, 0f, Hidden);
        Proj("video_patch_proj.weight", F32, PatchDim, Hidden);
        Add("video_patch_proj.bias", F32, 0.05f, 0f, Hidden);
        for (int b = 0; b < RefinerBlocks; b++) Block($"token_refiner.blocks.{b}.", modulated: false);
        Add("token_refiner.final_norm.weight", BF16, 0.2f, 1f, Hidden);
        for (int b = 0; b < Blocks; b++) Block($"blocks.{b}.", modulated: true);
        Add("final_layer.norm.weight", BF16, 0.2f, 1f, Hidden);
        Add("final_layer.adaln_proj.linear.weight", F16, 0.3f, 0f, TimeDim, 2 * Hidden);
        Add("final_layer.adaln_proj.linear.bias", BF16, 0.1f, 0f, 2 * Hidden);
        Proj("final_layer.video_out.weight", F32, Hidden, PddSteps * PatchDim);
        Add("final_layer.video_out.bias", F32, 0.05f, 0f, PddSteps * PatchDim);
        // Inverse frequencies as a RoPE's: 1 / 10000^(k / n).
        tensors.Add(new T("rope.inv_freq", new long[] { InvFreq }, F32,
            Enumerable.Range(0, InvFreq).Select(k => (float)Math.Pow(10000, -(double)k / InvFreq)).ToArray()));
        return tensors;
    }

    private static byte[] Encode(T t)
    {
        long n = t.Data.LongLength;
        switch (t.Type)
        {
            case F32:
                return MemoryMarshal.AsBytes(t.Data.AsSpan()).ToArray();
            case F16:
                return MemoryMarshal.AsBytes(t.Data.Select(v => BitConverter.HalfToUInt16Bits((Half)v)).ToArray().AsSpan()).ToArray();
            case BF16:
                return MemoryMarshal.AsBytes(t.Data.Select(Bf16).ToArray().AsSpan()).ToArray();
            default:
            {
                long rows = n / t.Shape[0];
                long blockSize = TensorSharp.Runtime.GgufFile.GetBlockSize((TensorSharp.Runtime.GgmlTensorType)t.Type);
                long typeSize = TensorSharp.Runtime.GgufFile.GetTypeSize((TensorSharp.Runtime.GgmlTensorType)t.Type);
                var bytes = new byte[t.Shape[0] / blockSize * typeSize * rows];
                GCHandle src = GCHandle.Alloc(t.Data, GCHandleType.Pinned), dst = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                try
                {
                    long written = GgmlGgufTensorDequant.QuantizeFloat32RowsOrZero(t.Type, src.AddrOfPinnedObject(), dst.AddrOfPinnedObject(), rows, t.Shape[0]);
                    if (written != bytes.LongLength) throw new InvalidOperationException($"quantizing {t.Name} wrote {written} of {bytes.LongLength} bytes");
                }
                finally
                {
                    src.Free();
                    dst.Free();
                }
                return bytes;
            }
        }
    }

    private static ushort Bf16(float v)
    {
        uint bits = BitConverter.SingleToUInt32Bits(v);
        bits += 0x7FFFu + ((bits >> 16) & 1u);   // round to nearest even
        return (ushort)(bits >> 16);
    }

    /// <summary>A safetensors file without metadata: the shapes in torch order (the GGUF's ne order reversed).</summary>
    private static void WriteSafetensors(string path, IReadOnlyList<T> tensors)
    {
        var header = new Dictionary<string, object>();
        byte[][] data = tensors.Select(Encode).ToArray();
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            string dtype = tensors[i].Type switch { F32 => "F32", F16 => "F16", BF16 => "BF16", _ => throw new ArgumentException(tensors[i].Name) };
            header[tensors[i].Name] = new Dictionary<string, object>
            {
                ["dtype"] = dtype,
                ["shape"] = tensors[i].Shape.Reverse().ToArray(),
                ["data_offsets"] = new[] { offset, offset + data[i].LongLength },
            };
            offset += data[i].LongLength;
        }
        byte[] json = Encoding.UTF8.GetBytes(System.Text.Json.JsonSerializer.Serialize(header));
        using var stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        foreach (byte[] bytes in data) stream.Write(bytes);
    }

    /// <summary>A GGUF v3 without metadata, 32-byte aligned.</summary>
    private static void Write(string path, IReadOnlyList<T> tensors)
    {
        static long Align(long value) => (value + 31) / 32 * 32;
        byte[][] data = tensors.Select(Encode).ToArray();
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)tensors.Count); writer.Write(0UL);
        long offset = 0;
        for (int i = 0; i < tensors.Count; i++)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(tensors[i].Name);
            writer.Write((ulong)utf8.Length); writer.Write(utf8); writer.Write((uint)tensors[i].Shape.Length);
            foreach (long size in tensors[i].Shape) writer.Write((ulong)size);
            writer.Write((uint)tensors[i].Type); writer.Write((ulong)offset);
            offset += Align(data[i].LongLength);
        }
        writer.Write(new byte[Align(stream.Position) - stream.Position]);
        foreach (byte[] bytes in data)
        {
            writer.Write(bytes);
            writer.Write(new byte[Align(bytes.LongLength) - bytes.LongLength]);
        }
    }
}
