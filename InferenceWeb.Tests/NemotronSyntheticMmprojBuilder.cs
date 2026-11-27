// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Builds a tiny, deterministic Nemotron vision mmproj (clip, RADIO ViT, nemotron_omni MLP
// projector) for NemotronVisionEncoder tests: 2 blocks of hidden 32 / 2 heads, a 64x64 tile
// of 16x16 patches, 2 class tokens and a 2x2 pixel shuffle, so one tile encodes to 4 tokens.
//
// The attention projection is written fused (v.blk.N.attn_qkv, as the Nano Omni and Super VL
// mmprojs store it) or split into attn_q/attn_k/attn_v weights and biases (as Bolt's does).
// The split tensors are row slices of the SAME name-seeded fused values, and the q/k/v type
// quantizes whole rows (F32, BF16, or Q8_0 with one 32-value block per row), so both layouts
// hold bit-identical weights.
using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

internal sealed class NemotronSyntheticMmprojBuilder
{
    public const int ImageSize = 64;
    public const int PatchSize = 16;
    public const int Channels = 3;
    public const int Hidden = 32;
    public const int Heads = 2;
    public const int Intermediate = 64;
    public const int Blocks = 2;
    public const int ScaleFactor = 2;
    public const int ClassTokens = 2;
    public const int ProjectionHidden = 48;

    /// <summary>Tokens one <see cref="ImageSize"/> tile encodes to.</summary>
    public const int TokensPerTile = ImageSize / PatchSize / ScaleFactor * (ImageSize / PatchSize / ScaleFactor);

    /// <summary>The projector's output width (mm.model.mlp.3); defaults to the synthetic
    /// Nemotron language model's hidden size.</summary>
    public int ProjectionDim { get; init; } = NemotronSyntheticModelBuilder.Hidden;

    /// <summary>Store attn_q/attn_k/attn_v (and their biases) instead of attn_qkv.</summary>
    public bool SplitQkv { get; init; }

    /// <summary>Store block 0's projection in BOTH layouts (a malformed file).</summary>
    public bool FusedAndSplitBlock0 { get; init; }

    /// <summary>The q/k/v weight type (the biases stay F32, as in the published files).</summary>
    public SyntheticGguf.GgmlType QkvType { get; init; } = SyntheticGguf.GgmlType.F32;

    /// <summary>Add the video tower's temporal patch embedding v.patch_embd.weight.1.</summary>
    public bool VideoPatchEmbedding { get; init; }

    /// <summary>Add the tower's final LayerNorm v.post_ln (weight and bias), as Super VL's mmproj stores it.</summary>
    public bool PostLn { get; init; }

    /// <summary>Tensor names to leave out of the file.</summary>
    public IReadOnlyCollection<string> Omit { get; init; } = Array.Empty<string>();

    /// <summary>The clip.vision.feed_forward_length to declare (the tensors stay <see cref="Intermediate"/> wide).</summary>
    public int DeclaredFeedForward { get; init; } = Intermediate;

    private static readonly string[] SplitModules = { "attn_q", "attn_k", "attn_v" };

    public string Write(string path)
    {
        int gridTokens = ImageSize / PatchSize * (ImageSize / PatchSize);
        int merged = Hidden * ScaleFactor * ScaleFactor;
        var tensors = new List<SyntheticGguf.Tensor>
        {
            // GGUF dims, ne0 first.
            SyntheticGguf.Gen("v.patch_embd.weight", 0.05f, PatchSize, PatchSize, Channels, Hidden),
            SyntheticGguf.Gen("v.patch_embd.bias", 0.05f, Hidden),
            SyntheticGguf.Gen("v.position_embd.weight", 0.1f, Hidden, gridTokens),
            SyntheticGguf.Gen("v.class_embd", 0.1f, Hidden, ClassTokens),
            SyntheticGguf.GenAround("mm.model.mlp.0.weight", 1f, 0.1f, merged),
            SyntheticGguf.Gen("mm.model.mlp.1.weight", 0.1f, merged, ProjectionHidden),
            SyntheticGguf.Gen("mm.model.mlp.3.weight", 0.1f, ProjectionHidden, ProjectionDim),
        };
        if (VideoPatchEmbedding)
            tensors.Add(SyntheticGguf.Gen("v.patch_embd.weight.1", 0.05f, PatchSize, PatchSize, 2 * Channels, Hidden));
        if (PostLn)
        {
            tensors.Add(SyntheticGguf.GenAround("v.post_ln.weight", 1f, 0.3f, Hidden));
            tensors.Add(SyntheticGguf.Gen("v.post_ln.bias", 0.2f, Hidden));
        }

        for (int b = 0; b < Blocks; b++)
        {
            string p = $"v.blk.{b}";
            tensors.Add(SyntheticGguf.GenAround($"{p}.ln1.weight", 1f, 0.1f, Hidden));
            tensors.Add(SyntheticGguf.Gen($"{p}.ln1.bias", 0.05f, Hidden));

            var qkvWeight = SyntheticGguf.Gen($"{p}.attn_qkv.weight", 0.15f, Hidden, 3 * Hidden);
            qkvWeight.Type = QkvType;
            var qkvBias = SyntheticGguf.Gen($"{p}.attn_qkv.bias", 0.05f, 3 * Hidden);
            bool split = SplitQkv || (FusedAndSplitBlock0 && b == 0);
            bool fused = !SplitQkv || (FusedAndSplitBlock0 && b == 0);
            if (fused)
            {
                tensors.Add(qkvWeight);
                tensors.Add(qkvBias);
            }
            if (split)
            {
                tensors.AddRange(SplitRows(qkvWeight, p, "weight"));
                tensors.AddRange(SplitRows(qkvBias, p, "bias"));
            }

            tensors.Add(SyntheticGguf.Gen($"{p}.attn_out.weight", 0.15f, Hidden, Hidden));
            tensors.Add(SyntheticGguf.Gen($"{p}.attn_out.bias", 0.05f, Hidden));
            tensors.Add(SyntheticGguf.GenAround($"{p}.ln2.weight", 1f, 0.1f, Hidden));
            tensors.Add(SyntheticGguf.Gen($"{p}.ln2.bias", 0.05f, Hidden));
            tensors.Add(SyntheticGguf.Gen($"{p}.ffn_up.weight", 0.15f, Hidden, Intermediate));
            tensors.Add(SyntheticGguf.Gen($"{p}.ffn_up.bias", 0.05f, Intermediate));
            tensors.Add(SyntheticGguf.Gen($"{p}.ffn_down.weight", 0.15f, Intermediate, Hidden));
            tensors.Add(SyntheticGguf.Gen($"{p}.ffn_down.bias", 0.05f, Hidden));
        }
        tensors.RemoveAll(t => Omit.Contains(t.Name));

        var kv = new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "clip" },
            new SyntheticGguf.Str { Key = "general.type", V = "mmproj" },
            new SyntheticGguf.Bool { Key = "clip.has_vision_encoder", V = true },
            new SyntheticGguf.Str { Key = "clip.projector_type", V = "nemotron_omni" },
            new SyntheticGguf.Bool { Key = "clip.use_gelu", V = true },
            new SyntheticGguf.U32 { Key = "clip.vision.embedding_length", V = Hidden },
            new SyntheticGguf.U32 { Key = "clip.vision.feed_forward_length", V = (uint)DeclaredFeedForward },
            new SyntheticGguf.U32 { Key = "clip.vision.attention.head_count", V = Heads },
            new SyntheticGguf.U32 { Key = "clip.vision.block_count", V = Blocks },
            new SyntheticGguf.U32 { Key = "clip.vision.image_size", V = ImageSize },
            new SyntheticGguf.U32 { Key = "clip.vision.patch_size", V = PatchSize },
            new SyntheticGguf.U32 { Key = "clip.vision.projection_dim", V = (uint)ProjectionDim },
            new SyntheticGguf.U32 { Key = "clip.vision.projector.scale_factor", V = ScaleFactor },
            new SyntheticGguf.F32 { Key = "clip.vision.attention.layer_norm_epsilon", V = 1e-6f },
            new SyntheticGguf.F32Arr { Key = "clip.vision.image_mean", V = new[] { 0.48145467f, 0.4578275f, 0.40821072f } },
            new SyntheticGguf.F32Arr { Key = "clip.vision.image_std", V = new[] { 0.26862955f, 0.2613026f, 0.2757771f } },
        };

        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    /// <summary>The fused tensor's Q, K and V output rows as attn_q/attn_k/attn_v
    /// (GGUF [in, 3h] -> three [in, h]; a bias [3h] -> three [h]).</summary>
    private static IEnumerable<SyntheticGguf.Tensor> SplitRows(SyntheticGguf.Tensor fused, string blockPrefix, string suffix)
    {
        bool matrix = fused.Dims.Length == 2;
        int rowLength = matrix ? (int)fused.Dims[0] : 1;
        int rows = (int)fused.Dims[^1] / SplitModules.Length;
        for (int part = 0; part < SplitModules.Length; part++)
        {
            var data = new float[rows * rowLength];
            Array.Copy(fused.Data, part * data.Length, data, 0, data.Length);
            yield return new SyntheticGguf.Tensor
            {
                Name = $"{blockPrefix}.{SplitModules[part]}.{suffix}",
                Dims = matrix ? new[] { (ulong)rowLength, (ulong)rows } : new[] { (ulong)rows },
                Data = data,
                Type = fused.Type,
            };
        }
    }
}
