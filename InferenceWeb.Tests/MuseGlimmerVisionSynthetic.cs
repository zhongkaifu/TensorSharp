// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace InferenceWeb.Tests;

/// <summary>
/// A small mmproj of the <c>muse-glimmer</c> projector type (Muse-Glimmer and Muse Spark), laid out as the real one
/// (unsloth's mmproj-Muse-Glimmer-30B-Q8_0.gguf): F32 norms, biases, patch kernel and position grid, Q8_0 products. Width
/// 64, 4 heads of 16, a 128-wide MLP, 5 layers (so layers 0-2 attend in windows and 3 and 4 globally), a 4 x 4 position
/// grid (windows of 4 x 4 patches), patch 14, merge 2, an adapter of 256 -> 96 -> 96 -> <c>projection</c>. The products can
/// be written F32 instead: executors whose Q8_0 products quantize their activations (ggml-cpu, the managed executor) leave
/// about 2% between two otherwise equal towers on these random weights, which would hide what the parity tests look for.
/// </summary>
internal static class MuseGlimmerVisionSynthetic
{
    internal const int Width = 64, Heads = 4, FeedForward = 128, Layers = 5, Side = 4, Patch = 14, AdapterWidth = 96;

    /// <summary>Write the mmproj to <paramref name="path"/>; <paramref name="projection"/> is the language model's width it
    /// feeds (<c>clip.vision.projection_dim</c>).</summary>
    internal static string Write(string path, int projection = 48, SyntheticGguf.GgmlType products = SyntheticGguf.GgmlType.Q8_0)
    {
        s_products = products;
        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "clip" },
            new SyntheticGguf.Bool { Key = "clip.has_vision_encoder", V = true },
            new SyntheticGguf.Str { Key = "clip.projector_type", V = "muse-glimmer" },
            new SyntheticGguf.U32 { Key = "clip.vision.projection_dim", V = (uint)projection },
            new SyntheticGguf.U32 { Key = "clip.vision.image_size", V = (uint)(Side * Patch) },
            new SyntheticGguf.U32 { Key = "clip.vision.patch_size", V = Patch },
            new SyntheticGguf.U32 { Key = "clip.vision.embedding_length", V = Width },
            new SyntheticGguf.U32 { Key = "clip.vision.feed_forward_length", V = FeedForward },
            new SyntheticGguf.U32 { Key = "clip.vision.block_count", V = Layers },
            new SyntheticGguf.U32 { Key = "clip.vision.attention.head_count", V = Heads },
            new SyntheticGguf.F32 { Key = "clip.vision.attention.layer_norm_epsilon", V = 1e-5f },
            new SyntheticGguf.U32 { Key = "clip.vision.spatial_merge_size", V = 2 },
            new SyntheticGguf.F32Arr { Key = "clip.vision.image_mean", V = new[] { 0.5f, 0.5f, 0.5f } },
            new SyntheticGguf.F32Arr { Key = "clip.vision.image_std", V = new[] { 0.5f, 0.5f, 0.5f } },
        };
        var t = new List<SyntheticGguf.Tensor>
        {
            SyntheticGguf.Gen("v.patch_embd.weight", 0.08f, Patch, Patch, 3, Width),
            SyntheticGguf.Gen("v.position_embd.weight", 0.5f, Width, Side * Side),
        };
        Norm(t, "v.pre_ln");
        for (int l = 0; l < Layers; l++)
        {
            string p = $"v.blk.{l}.";
            Norm(t, p + "ln1");
            Norm(t, p + "ln2");
            foreach (string n in new[] { "attn_q", "attn_k", "attn_v", "attn_out" })
            {
                t.Add(Q8(p + n + ".weight", 0.18f, Width, Width));
                t.Add(SyntheticGguf.Gen(p + n + ".bias", 0.05f, Width));
            }
            t.Add(Q8(p + "ffn_up.weight", 0.15f, Width, FeedForward));
            t.Add(SyntheticGguf.Gen(p + "ffn_up.bias", 0.05f, FeedForward));
            t.Add(Q8(p + "ffn_down.weight", 0.1f, FeedForward, Width));
            t.Add(SyntheticGguf.Gen(p + "ffn_down.bias", 0.05f, Width));
        }
        Norm(t, "v.post_ln");
        t.Add(Q8("mm.0.weight", 0.08f, 4 * Width, AdapterWidth));
        t.Add(Q8("mm.1.weight", 0.12f, AdapterWidth, AdapterWidth));
        t.Add(Q8("mm.2.weight", 0.12f, AdapterWidth, projection));
        SyntheticGguf.Write(path, kv, t);
        return path;
    }

    private static void Norm(List<SyntheticGguf.Tensor> t, string prefix)
    {
        t.Add(SyntheticGguf.GenAround(prefix + ".weight", 1f, 0.1f, Width));
        t.Add(SyntheticGguf.Gen(prefix + ".bias", 0.05f, Width));
    }

    [ThreadStatic] private static SyntheticGguf.GgmlType s_products;

    /// <summary>A product weight in the format <see cref="Write"/> was asked for (Q8_0 as the real mmproj, or F32).</summary>
    private static SyntheticGguf.Tensor Q8(string name, float scale, params int[] dims)
    {
        SyntheticGguf.Tensor w = SyntheticGguf.Gen(name, scale, dims);
        w.Type = s_products;
        return w;
    }

    /// <summary>Channel-first [3, height, width] pixels in [-1, 1] (the processor's 0.5 / 0.5 normalization's range): smooth
    /// gradients, a ring and seeded noise, so neighbouring patches differ and the position grid matters.</summary>
    internal static float[] Pixels(int width, int height, uint seed)
    {
        var px = new float[3L * width * height];
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    double u = (double)x / width, v = (double)y / height;
                    double ring = Math.Sin(9 * Math.Sqrt((u - 0.3) * (u - 0.3) + (v - 0.6) * (v - 0.6)) + c);
                    double value = 0.5 * Math.Sin(4 * u + 2 * c) * Math.Cos(3 * v - c) + 0.3 * ring + 0.15 * Noise(ref state);
                    px[((long)c * height + y) * width + x] = (float)Math.Clamp(value, -1, 1);
                }
        return px;
    }

    /// <summary>xorshift32 as a value in [-1, 1).</summary>
    private static double Noise(ref uint s)
    {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s / 2147483648.0 - 1.0;
    }
}
