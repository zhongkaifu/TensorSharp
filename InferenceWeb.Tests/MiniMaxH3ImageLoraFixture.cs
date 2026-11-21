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
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

/// <summary>
/// Tiny MiniMax-H3-Image LoRA adapters for <see cref="MiniMaxH3ImageFixture"/>'s transformer (hidden 256, 4 x 64 heads, FFN
/// 512, 2 blocks and 1 token-refiner block), written at test time from a seed in any of the three key grammars
/// (docs/design/diffusion-runtime.md section 5.3). A term is drawn once in the checkpoint's own layout (qkv stacked
/// [q; k; v], fc1 [gate; value]), so the same update can be written as diffusers PEFT (q, k and v apart, ff.net.0.proj's
/// halves swapped to [value; gate]), ComfyUI native (fused qkv; q, k and v's terms as one block-diagonal B over their
/// concatenated A, alpha x 3) or kohya, and merged into a reference checkpoint independently of the loader under test
/// (<see cref="Deltas"/>).
/// </summary>
internal sealed class MiniMaxH3ImageLoraFixture : IDisposable
{
    internal const int Hidden = MiniMaxH3ImageFixture.Hidden, Inner = MiniMaxH3ImageFixture.Inner, Ffn = MiniMaxH3ImageFixture.Ffn;
    internal const int Blocks = MiniMaxH3ImageFixture.Blocks, RefinerBlocks = MiniMaxH3ImageFixture.RefinerBlocks;

    internal enum Proj { Qkv, Out, Fc1, Fc2 }

    internal enum Grammar { Diffusers, Comfy, Kohya }

    /// <summary>One update in the checkpoint's layout: A [rank, in] and B [out, rank] row-major, applied at
    /// <see cref="Scale"/> (alpha / rank). <see cref="PerThird"/> (qkv only): q, k and v each have their own <see cref="Rank"/>
    /// components, A is [3 rank, in] and B block-diagonal; else every component reaches all of qkv.</summary>
    internal sealed record Term(bool Refiner, int Block, Proj P, int Rank, float[] A, float[] B, bool PerThird, float Scale);

    internal MiniMaxH3ImageLoraFixture()
    {
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3img-lora-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
    }

    internal string Dir { get; }

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
    }

    internal static (int In, int Out) Shape(Proj p) => p switch
    {
        Proj.Qkv => (Hidden, 3 * Inner),
        Proj.Out => (Inner, Hidden),
        Proj.Fc1 => (Hidden, 2 * Ffn),
        _ => (Ffn, Hidden),
    };

    /// <summary>The fixture tensor a term updates.</summary>
    internal static string WeightName(bool refiner, int block, Proj p) =>
        $"{(refiner ? "token_refiner.blocks" : "blocks")}.{block}." + p switch
        {
            Proj.Qkv => "attn.qkv_proj.weight",
            Proj.Out => "attn.out_proj.weight",
            Proj.Fc1 => "mlp.fc1.weight",
            _ => "mlp.fc2.weight",
        };

    /// <summary>Terms on <paramref name="projections"/> of every block (refiner included unless <paramref name="trunkOnly"/>),
    /// with values about <paramref name="magnitude"/> / sqrt(in) on both factors.</summary>
    internal static List<Term> Random(int seed, int rank, float scale, bool perThirdQkv, float magnitude = 1f, bool trunkOnly = false,
        Proj[] projections = null)
    {
        var rng = new Random(seed);
        float[] Values(long n, float s) => Enumerable.Range(0, checked((int)n)).Select(_ => s * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var terms = new List<Term>();
        projections ??= new[] { Proj.Qkv, Proj.Out, Proj.Fc1, Proj.Fc2 };
        foreach (bool refiner in trunkOnly ? new[] { false } : new[] { true, false })
            for (int b = 0; b < (refiner ? RefinerBlocks : Blocks); b++)
                foreach (Proj p in projections)
                {
                    var (input, output) = Shape(p);
                    float sa = magnitude / MathF.Sqrt(input), sb = magnitude / MathF.Sqrt(rank);
                    if (p == Proj.Qkv && perThirdQkv)
                    {
                        float[] a = Values(3L * rank * input, sa);
                        var bm = new float[(long)output * 3 * rank];
                        for (int t = 0; t < 3; t++)
                            for (long o = 0; o < Inner; o++)
                                for (int k = 0; k < rank; k++)
                                    bm[(t * Inner + o) * 3 * rank + t * rank + k] = sb * (float)(rng.NextDouble() * 2 - 1);
                        terms.Add(new Term(refiner, b, p, rank, a, bm, true, scale));
                    }
                    else
                        terms.Add(new Term(refiner, b, p, rank, Values((long)rank * input, sa), Values((long)output * rank, sb), false, scale));
                }
        return terms;
    }

    /// <summary>strength * scale * B A of every term, summed per fixture tensor ([out, in] row-major): the merge a reference
    /// checkpoint adds to its projections.</summary>
    internal static Dictionary<string, float[]> Deltas(IEnumerable<(IReadOnlyList<Term> Terms, float Strength)> adapters)
    {
        var deltas = new Dictionary<string, float[]>(StringComparer.Ordinal);
        foreach (var (terms, strength) in adapters)
            foreach (Term t in terms)
            {
                var (input, output) = Shape(t.P);
                string name = WeightName(t.Refiner, t.Block, t.P);
                if (!deltas.TryGetValue(name, out float[] d)) deltas[name] = d = new float[(long)output * input];
                int rank = t.PerThird ? 3 * t.Rank : t.Rank;
                double s = (double)strength * t.Scale;
                for (long o = 0; o < output; o++)
                    for (long i = 0; i < input; i++)
                    {
                        double sum = 0;
                        for (int k = 0; k < rank; k++) sum += (double)t.B[o * rank + k] * t.A[k * input + i];
                        d[o * input + i] += (float)(s * sum);
                    }
            }
        return deltas;
    }

    /// <summary>A <see cref="MiniMaxH3ImageFixture"/> edit that adds <paramref name="deltas"/> to its projections.</summary>
    internal static Func<string, float[], float[]> Merge(Dictionary<string, float[]> deltas) => (name, data) =>
    {
        if (!deltas.TryGetValue(name, out float[] d)) return data;
        for (long i = 0; i < data.LongLength; i++) data[i] += d[i];
        return data;
    };

    /// <summary>The tensors and metadata of <paramref name="terms"/> in <paramref name="grammar"/>.</summary>
    internal sealed class File
    {
        public readonly List<(string Name, long[] Shape, float[] Data, string Dtype)> Tensors = new();
        public readonly Dictionary<string, string> Metadata = new(StringComparer.Ordinal);
    }

    /// <param name="alpha">Write the alpha (diffusers: the "alpha" metadata; ComfyUI and kohya: a .alpha tensor a module);
    /// false leaves it to the config or the rank.</param>
    /// <param name="downUp">ComfyUI: lora_down / lora_up instead of lora_A / lora_B.</param>
    /// <param name="prefix">diffusers: "transformer."; ComfyUI: "diffusion_model."; or "" for none.</param>
    internal static File Build(IReadOnlyList<Term> terms, Grammar grammar, bool alpha = true, string dtype = "F32", bool downUp = false,
        string prefix = null, string slot = null)
    {
        var f = new File();
        float? diffusersScale = null;
        foreach (Term t in terms)
        {
            var (input, output) = Shape(t.P);
            switch (grammar)
            {
                case Grammar.Diffusers:
                {
                    string scope = $"{prefix ?? ""}{(t.Refiner ? "token_refiner.refiner_blocks" : "transformer_blocks")}.{t.Block}.";
                    string tail = slot == null ? ".weight" : $".{slot}.weight";
                    void Add(string module, float[] a, float[] b, long rank, long rows)
                    {
                        f.Tensors.Add((scope + module + ".lora_A" + tail, new[] { rank, (long)input }, a, dtype));
                        f.Tensors.Add((scope + module + ".lora_B" + tail, new[] { rows, rank }, b, dtype));
                    }
                    if (diffusersScale is { } ds && ds != t.Scale) throw new ArgumentException("diffusers metadata holds one alpha: one scale for all terms");
                    diffusersScale = t.Scale;
                    if (t.P == Proj.Qkv)
                    {
                        int rank = t.PerThird ? t.Rank : t.Rank;
                        string[] modules = { "attn.to_q", "attn.to_k", "attn.to_v" };
                        for (int s = 0; s < 3; s++)
                        {
                            float[] a = t.PerThird ? t.A.Skip(s * t.Rank * input).Take(t.Rank * input).ToArray() : t.A;
                            int width = t.PerThird ? 3 * t.Rank : t.Rank;
                            var b = new float[(long)Inner * rank];
                            for (long o = 0; o < Inner; o++)
                                for (int k = 0; k < rank; k++)
                                    b[o * rank + k] = t.B[(s * Inner + o) * width + (t.PerThird ? s * t.Rank + k : k)];
                            Add(modules[s], a, b, rank, Inner);
                        }
                    }
                    else if (t.P == Proj.Fc1)
                    {
                        // diffusers' ff.net.0.proj reads [value; gate]: the native [gate; value] rows swapped.
                        var b = new float[t.B.Length];
                        long half = (long)Ffn * t.Rank;
                        Array.Copy(t.B, half, b, 0, half);
                        Array.Copy(t.B, 0, b, half, half);
                        Add("ff.net.0.proj", t.A, b, t.Rank, output);
                    }
                    else
                        Add(t.P == Proj.Out ? "attn.to_out.0" : "ff.net.2", t.A, t.B, t.Rank, output);
                    break;
                }
                default:
                {
                    bool kohya = grammar == Grammar.Kohya;
                    string module = kohya
                        ? $"lora_unet_{(t.Refiner ? "token_refiner_blocks" : "blocks")}_{t.Block}_" +
                          t.P switch { Proj.Qkv => "attn_qkv_proj", Proj.Out => "attn_out_proj", Proj.Fc1 => "mlp_fc1", _ => "mlp_fc2" }
                        : $"{prefix ?? "diffusion_model."}{(t.Refiner ? "token_refiner.blocks" : "blocks")}.{t.Block}." +
                          t.P switch { Proj.Qkv => "attn.qkv_proj", Proj.Out => "attn.out_proj", Proj.Fc1 => "mlp.fc1", _ => "mlp.fc2" };
                    int rank = t.PerThird ? 3 * t.Rank : t.Rank;
                    string down = kohya || downUp ? ".lora_down.weight" : ".lora_A.weight", up = kohya || downUp ? ".lora_up.weight" : ".lora_B.weight";
                    f.Tensors.Add((module + down, new[] { (long)rank, input }, t.A, dtype));
                    f.Tensors.Add((module + up, new[] { (long)output, rank }, t.B, dtype));
                    if (alpha) f.Tensors.Add((module + ".alpha", Array.Empty<long>(), new[] { t.Scale * rank }, "F32"));
                    break;
                }
            }
        }
        if (grammar == Grammar.Diffusers && alpha && diffusersScale is { } scale)
        {
            int rank = terms[0].Rank;
            f.Metadata["alpha"] = (scale * rank).ToString(System.Globalization.CultureInfo.InvariantCulture);
        }
        return f;
    }

    /// <summary>Write <paramref name="file"/> as safetensors in <see cref="Dir"/>; returns its path.</summary>
    internal string Write(File file, string fileName)
    {
        string path = Path.Combine(Dir, fileName);
        var header = new JsonObject();
        if (file.Metadata.Count > 0)
        {
            var meta = new JsonObject();
            foreach (var (k, v) in file.Metadata) meta[k] = v;
            header["__metadata__"] = meta;
        }
        var blobs = new List<byte[]>();
        long offset = 0;
        foreach (var (tensor, shape, data, dtype) in file.Tensors)
        {
            byte[] blob = MiniMaxH3LoraFixtures.Encode(data, dtype);
            var dims = new JsonArray();
            foreach (long d in shape) dims.Add(d);
            header[tensor] = new JsonObject { ["dtype"] = dtype, ["shape"] = dims, ["data_offsets"] = new JsonArray(offset, offset + blob.Length) };
            offset += blob.Length;
            blobs.Add(blob);
        }
        byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = System.IO.File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)json.Length));
        fs.Write(json);
        foreach (byte[] blob in blobs) fs.Write(blob);
        return path;
    }

    /// <summary>Write a <c>minimax-h3-image-lora</c> config naming <paramref name="weights"/> beside it; a non-null
    /// <paramref name="steps"/> makes it a speed adapter on <paramref name="sigmas"/> with <paramref name="heads"/>.</summary>
    internal string WriteConfig(string fileName, string weights, string name = "Test adapter", float? scale = null, float? alpha = null,
        string keyFormat = null, int? steps = null, float[] sigmas = null, string heads = "by-step", float cfg = 1f, long floor = 1024)
    {
        var root = new JsonObject { ["type"] = TensorSharp.Models.MiniMaxH3Image.MiniMaxH3ImageLoraConfig.Type, ["name"] = name, ["weights"] = Path.GetFileName(weights) };
        if (keyFormat != null) root["key_format"] = keyFormat;
        if (alpha is { } a) root["alpha"] = a;
        if (scale is { } s) root["scale"] = s;
        if (steps is { } n)
        {
            var list = new JsonArray();
            foreach (float v in sigmas ?? new[] { 1f, 12f / 13f, 0f }) list.Add(v);
            root["sampling"] = new JsonObject { ["steps"] = n, ["sigmas"] = list, ["heads"] = heads, ["cfg"] = cfg, ["min_generation_area"] = floor };
        }
        string path = Path.Combine(Dir, fileName);
        System.IO.File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return path;
    }
}
