// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// Writes the smallest <c>qwen3vl_merger</c> projector GGUF the Qwen-VL vision
/// encoder accepts: one block, hidden 8, two heads, patch 16, merge 2, and both
/// temporal patch-embedding slices (<c>v.patch_embd.weight</c> / <c>.weight.1</c>),
/// which is what makes it a VIDEO-capable tower rather than a still-image one.
/// The projection width is 8 so the embeddings fit the synthetic Qwen4Exp target
/// fixture (embedding_length 8). Weights are deterministic pseudo-random values:
/// this is a wiring fixture for temporal merging, placeholder expansion and
/// position assignment, not an encoder-quality fixture.
/// </summary>
internal static class QwenVLSyntheticMmprojBuilder
{
    public const int ImageSize = 32;
    public const int PatchSize = 16;
    public const int Hidden = 8;
    public const int Intermediate = 16;
    public const int Heads = 2;
    public const int Blocks = 1;
    public const int ProjectionDim = 8;
    public const int MergeSize = 2;

    private const int Alignment = 32;

    public static string Write(string path)
    {
        int gridPerSide = ImageSize / PatchSize;
        int mergedHidden = Hidden * MergeSize * MergeSize;
        var tensors = new List<TensorSpec>
        {
            // GGUF dims are ne0-first (the reverse of the row-major managed shape).
            Gen("v.patch_embd.weight", 0.05f, PatchSize, PatchSize, 3, Hidden),
            Gen("v.patch_embd.weight.1", 0.05f, PatchSize, PatchSize, 3, Hidden),
            Gen("v.patch_embd.bias", 0.05f, Hidden),
            Gen("v.position_embd.weight", 0.1f, Hidden, gridPerSide * gridPerSide),
            Gen("v.post_ln.weight", 0.1f, Hidden, offset: 1f),
            Gen("v.post_ln.bias", 0.05f, Hidden),
            Gen("mm.0.weight", 0.1f, mergedHidden, mergedHidden),
            Gen("mm.0.bias", 0.05f, mergedHidden),
            Gen("mm.2.weight", 0.1f, mergedHidden, ProjectionDim),
            Gen("mm.2.bias", 0.05f, ProjectionDim),
        };
        for (int b = 0; b < Blocks; b++)
        {
            string p = $"v.blk.{b}";
            tensors.Add(Gen($"{p}.ln1.weight", 0.1f, Hidden, offset: 1f));
            tensors.Add(Gen($"{p}.ln1.bias", 0.05f, Hidden));
            tensors.Add(Gen($"{p}.attn_qkv.weight", 0.1f, Hidden, 3 * Hidden));
            tensors.Add(Gen($"{p}.attn_qkv.bias", 0.05f, 3 * Hidden));
            tensors.Add(Gen($"{p}.attn_out.weight", 0.1f, Hidden, Hidden));
            tensors.Add(Gen($"{p}.attn_out.bias", 0.05f, Hidden));
            tensors.Add(Gen($"{p}.ln2.weight", 0.1f, Hidden, offset: 1f));
            tensors.Add(Gen($"{p}.ln2.bias", 0.05f, Hidden));
            tensors.Add(Gen($"{p}.ffn_up.weight", 0.1f, Hidden, Intermediate));
            tensors.Add(Gen($"{p}.ffn_up.bias", 0.05f, Intermediate));
            tensors.Add(Gen($"{p}.ffn_down.weight", 0.1f, Intermediate, Hidden));
            tensors.Add(Gen($"{p}.ffn_down.bias", 0.05f, Hidden));
        }

        var kv = new List<KvEntry>
        {
            new KvStr { Key = "general.architecture", V = "clip" },
            new KvStr { Key = "general.type", V = "mmproj" },
            new KvStr { Key = "general.name", V = "tiny-qwen3vl-merger-video" },
            new KvBool { Key = "clip.has_vision_encoder", V = true },
            new KvStr { Key = "clip.projector_type", V = "qwen3vl_merger" },
            new KvBool { Key = "clip.use_gelu", V = true },
            new KvU32 { Key = "clip.vision.image_size", V = ImageSize },
            new KvU32 { Key = "clip.vision.patch_size", V = PatchSize },
            new KvU32 { Key = "clip.vision.embedding_length", V = Hidden },
            new KvU32 { Key = "clip.vision.feed_forward_length", V = Intermediate },
            new KvU32 { Key = "clip.vision.attention.head_count", V = Heads },
            new KvU32 { Key = "clip.vision.block_count", V = Blocks },
            new KvU32 { Key = "clip.vision.projection_dim", V = ProjectionDim },
            new KvU32 { Key = "clip.vision.spatial_merge_size", V = MergeSize },
            new KvF32 { Key = "clip.vision.attention.layer_norm_epsilon", V = 1e-6f },
            new KvF32 { Key = "clip.vision.rope.freq_base", V = 10000f },
            new KvU32 { Key = "general.file_type", V = 0 },
            new KvU32 { Key = "general.quantization_version", V = 2 },
        };

        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    private sealed class TensorSpec
    {
        public string Name = "";
        public ulong[] Dims = Array.Empty<ulong>();
        public float[] Data = Array.Empty<float>();
    }

    private static TensorSpec Gen(string name, float scale, params int[] dims) => Gen(name, scale, 0f, dims);

    private static TensorSpec Gen(string name, float scale, int dim, float offset) => Gen(name, scale, offset, new[] { dim });

    private static TensorSpec Gen(string name, float scale, float offset, int[] dims)
    {
        long n = 1;
        foreach (int d in dims) n *= d;
        uint state = Fnv1a(name) | 1u;
        var data = new float[n];
        for (long i = 0; i < n; i++)
        {
            // xorshift32, mapped to a roughly symmetric value in [-scale, scale].
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 0xFFFF) / 32767.5f - 1f);
        }
        var ul = new ulong[dims.Length];
        for (int i = 0; i < dims.Length; i++) ul[i] = (ulong)dims[i];
        return new TensorSpec { Name = name, Dims = ul, Data = data };
    }

    private static uint Fnv1a(string text)
    {
        uint h = 2166136261;
        foreach (char c in text) { h ^= c; h *= 16777619; }
        return h;
    }

    private abstract class KvEntry { public string Key = ""; public abstract void Write(BinaryWriter w); }
    private sealed class KvU32 : KvEntry { public uint V; public override void Write(BinaryWriter w) { w.Write((uint)4); w.Write(V); } }
    private sealed class KvF32 : KvEntry { public float V; public override void Write(BinaryWriter w) { w.Write((uint)6); w.Write(V); } }
    private sealed class KvBool : KvEntry { public bool V; public override void Write(BinaryWriter w) { w.Write((uint)7); w.Write(V); } }
    private sealed class KvStr : KvEntry { public string V = ""; public override void Write(BinaryWriter w) { w.Write((uint)8); WriteStr(w, V); } }

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    private static void WriteGguf(Stream fs, List<KvEntry> kv, List<TensorSpec> tensors)
    {
        using var head = new MemoryStream();
        using (var w = new BinaryWriter(head, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);            // "GGUF"
            w.Write(3u);                      // version
            w.Write((ulong)tensors.Count);
            w.Write((ulong)kv.Count);
            foreach (var e in kv)
            {
                WriteStr(w, e.Key);
                e.Write(w);
            }
            ulong offset = 0;
            foreach (var t in tensors)
            {
                WriteStr(w, t.Name);
                w.Write((uint)t.Dims.Length);
                foreach (ulong d in t.Dims) w.Write(d);
                w.Write(0u);                  // GGML_TYPE_F32
                w.Write(offset);
                ulong bytes = (ulong)t.Data.Length * sizeof(float);
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        for (int i = 0; i < pad; i++) fs.WriteByte(0);

        var zeros = new byte[Alignment];
        foreach (var t in tensors)
        {
            byte[] raw = new byte[t.Data.Length * sizeof(float)];
            Buffer.BlockCopy(t.Data, 0, raw, 0, raw.Length);
            fs.Write(raw, 0, raw.Length);
            int tailPad = (Alignment - raw.Length % Alignment) % Alignment;
            if (tailPad > 0) fs.Write(zeros, 0, tailPad);
        }
    }
}
