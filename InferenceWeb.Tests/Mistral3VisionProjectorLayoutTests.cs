// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The catalog's Mistral Small 3.1 projector (bartowski's
// mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf) is a llama.cpp clip
// file: v.patch_embd / v.pre_ln / v.blk.N.ln1 / mm.patch_merger / mm.1 / mm.2. The
// encoder read only Ollama's names, so every image request threw
// KeyNotFoundException('v.patch_conv.weight') and the server answered HTTP 500.
// Past the 500 the model was still blind to the image (it read a red "4821" card as
// a blue "2975"). Against a transcription of HF's Pixtral tower on the real mmproj the
// encoder differed in four places, each pinned here: llama.cpp stores the vision Q/K
// rows permuted into interleaved RoPE pairs; the tower is GELU-gated (clip.use_gelu)
// where the encoder hardcoded SiLU; the 2D RoPE table was filled frequency-major but
// read patch-major; and the merge window was gathered patch-major where the merging
// layer expects torch unfold's channel-major order. And the injector copied the
// patch rows contiguously over a span whose tokens interleave [IMG_BREAK] markers.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public class Mistral3VisionProjectorLayoutTests : IDisposable
{
    private const int Patch = 4;
    private const int Hidden = 8;
    private const int Heads = 2;
    private const int Ffn = 16;
    private const int Projection = 12;
    private const int Merge = 2;

    private readonly string _dir;
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);

    public Mistral3VisionProjectorLayoutTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-mistral3-mmproj-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private enum Layout { LlamaCpp, Ollama }

    /// <summary>Same deterministic weights under either naming layout: each tensor's
    /// values are seeded by its Ollama name, so the two files differ only in names.</summary>
    private string WriteProjector(Layout layout, string fileName, Func<string, bool> omit = null, string activationKey = null)
    {
        var tensors = new List<(string Name, ulong[] Dims, float[] Data)>();
        void Add(string ollama, string llama, float scale, float offset, params int[] dims)
        {
            string name = layout == Layout.Ollama ? ollama : llama;
            if (omit != null && omit(name)) return;
            float[] data = Gen(ollama, dims.Aggregate(1, (a, b) => a * b), scale, offset);
            // llama.cpp's converter applies LlamaModel.permute to the vision Q/K rows.
            if (layout == Layout.LlamaCpp && (ollama.EndsWith("attn_q.weight") || ollama.EndsWith("attn_k.weight")))
                data = PermuteLikeLlamaCpp(data, Heads, Hidden / Heads);
            tensors.Add((name, dims.Select(d => (ulong)d).ToArray(), data));
        }

        Add("v.patch_conv.weight", "v.patch_embd.weight", 0.2f, 0f, Patch, Patch, 3, Hidden);
        Add("v.encoder_norm.weight", "v.pre_ln.weight", 0.2f, 1f, Hidden);
        const string b = "v.blk.0.";
        Add(b + "attn_norm.weight", b + "ln1.weight", 0.2f, 1f, Hidden);
        Add(b + "attn_q.weight", b + "attn_q.weight", 0.2f, 0f, Hidden, Hidden);
        Add(b + "attn_k.weight", b + "attn_k.weight", 0.2f, 0f, Hidden, Hidden);
        Add(b + "attn_v.weight", b + "attn_v.weight", 0.2f, 0f, Hidden, Hidden);
        Add(b + "attn_output.weight", b + "attn_out.weight", 0.2f, 0f, Hidden, Hidden);
        Add(b + "ffn_norm.weight", b + "ln2.weight", 0.2f, 1f, Hidden);
        Add(b + "ffn_gate.weight", b + "ffn_gate.weight", 0.2f, 0f, Hidden, Ffn);
        Add(b + "ffn_up.weight", b + "ffn_up.weight", 0.2f, 0f, Hidden, Ffn);
        Add(b + "ffn_down.weight", b + "ffn_down.weight", 0.2f, 0f, Ffn, Hidden);
        Add("mm.norm.weight", "mm.input_norm.weight", 0.2f, 1f, Hidden);
        Add("mm.patch_merger.merging_layer.weight", "mm.patch_merger.weight", 0.2f, 0f, Hidden * Merge * Merge, Hidden);
        Add("mm.linear_1.weight", "mm.1.weight", 0.2f, 0f, Hidden, Projection);
        Add("mm.linear_2.weight", "mm.2.weight", 0.2f, 0f, Projection, Projection);
        if (layout == Layout.LlamaCpp)
            tensors.Add(("v.token_embd.img_break", new ulong[] { Projection }, Gen("img_break", Projection, 0.2f, 0f)));

        var kv = new List<(string, Action<BinaryWriter>)>
        {
            ("general.architecture", Str("clip")),
            ("clip.projector_type", Str("pixtral")),
            ("clip.vision.image_size", U32(64)),
            ("clip.vision.patch_size", U32(Patch)),
            ("clip.vision.embedding_length", U32(Hidden)),
            ("clip.vision.feed_forward_length", U32(Ffn)),
            ("clip.vision.block_count", U32(1)),
            ("clip.vision.attention.head_count", U32(Heads)),
            ("clip.vision.attention.layer_norm_epsilon", F32(1e-5f)),
            ("clip.vision.spatial_merge_size", U32(Merge)),
        };
        activationKey ??= layout == Layout.LlamaCpp ? "clip.use_gelu" : null;
        if (activationKey != null)
            kv.Add((activationKey, Bool(true)));
        string path = Path.Combine(_dir, fileName);
        using var fs = File.Create(path);
        WriteGguf(fs, kv, tensors);
        return path;
    }

    private static float[] Pixels(int w, int h)
    {
        var p = new float[3 * w * h];
        for (int i = 0; i < p.Length; i++)
            p[i] = MathF.Sin(i * 0.37f) * 0.8f;
        return p;
    }

    [Fact]
    public void LlamaCppClipProjector_Loads_AndEncodesExactlyLikeTheOllamaLayout()
    {
        const int w = 4 * Patch, h = 2 * Patch; // 4x2 patches -> 2x1 merged tokens
        float[] pixels = Pixels(w, h);

        float[] ollama;
        using (var enc = new Mistral3VisionEncoder(WriteProjector(Layout.Ollama, "ollama.gguf"), _allocator))
        using (var t = enc.Encode(pixels, w, h))
            ollama = t.GetElementsAsFloat((int)t.ElementCount());

        using var llamaEnc = new Mistral3VisionEncoder(WriteProjector(Layout.LlamaCpp, "llamacpp.gguf"), _allocator);
        using var llama = llamaEnc.Encode(pixels, w, h);
        Assert.Equal(new long[] { 2, Projection }, llama.Sizes);
        float[] got = llama.GetElementsAsFloat((int)llama.ElementCount());
        Assert.Contains(got, v => v != 0f);
        Assert.All(got, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(ollama, got);
    }

    /// <summary>convert_hf_to_gguf.py LlamaModel.permute: reshape(n_head, 2, d/2, in),
    /// swap axes 1 and 2 - file row 2j + half is HF row half * d/2 + j.</summary>
    private static float[] PermuteLikeLlamaCpp(float[] w, int heads, int headDim)
    {
        int cols = w.Length / (heads * headDim);
        var r = new float[w.Length];
        for (int h = 0; h < heads; h++)
            for (int j = 0; j < headDim / 2; j++)
                for (int half = 0; half < 2; half++)
                    Array.Copy(w, (h * headDim + half * headDim / 2 + j) * cols, r, (h * headDim + 2 * j + half) * cols, cols);
        return r;
    }

    [Fact]
    public void VisionMlpActivation_FollowsTheFile_AndDefaultsToPixtralsGelu()
    {
        using (var enc = new Mistral3VisionEncoder(WriteProjector(Layout.LlamaCpp, "gelu.gguf"), _allocator))
            Assert.Equal(Mistral3VisionEncoder.VisionFfnActivation.Gelu, enc.FfnActivation);
        using (var enc = new Mistral3VisionEncoder(WriteProjector(Layout.LlamaCpp, "silu.gguf", activationKey: "clip.use_silu"), _allocator))
            Assert.Equal(Mistral3VisionEncoder.VisionFfnActivation.Silu, enc.FfnActivation);
        using (var enc = new Mistral3VisionEncoder(WriteProjector(Layout.Ollama, "nokey.gguf"), _allocator))
            Assert.Equal(Mistral3VisionEncoder.VisionFfnActivation.Gelu, enc.FfnActivation);

        // And the activation reaches the graph: the same weights under SiLU encode differently.
        const int w = 4 * Patch, h = 2 * Patch;
        float[] pixels = Pixels(w, h);
        using var gelu = new Mistral3VisionEncoder(Path.Combine(_dir, "gelu.gguf"), _allocator);
        using var silu = new Mistral3VisionEncoder(Path.Combine(_dir, "silu.gguf"), _allocator);
        using var a = gelu.Encode(pixels, w, h);
        using var b = silu.Encode(pixels, w, h);
        Assert.NotEqual(a.GetElementsAsFloat((int)a.ElementCount()), b.GetElementsAsFloat((int)b.ElementCount()));
    }

    /// <summary>HF PixtralRotaryEmbedding, transcribed: inv_freq[h * maxW + w] =
    /// cat(h * freqs[::2], w * freqs[1::2]), then cat(inv_freq, inv_freq). The encoder's
    /// table used to be filled frequency-major and read patch-major.</summary>
    [Fact]
    public void VisionRopeAngles_MatchPixtralRotaryEmbedding_PerPatch()
    {
        const int pw = 5, ph = 3, headDim = 16;
        const float theta = 10000f;
        float[] got = Mistral3VisionEncoder.BuildVisionRopeAngles(pw, ph, headDim, theta);
        Assert.Equal(pw * ph * headDim, got.Length);

        var freqs = Enumerable.Range(0, headDim / 2).Select(i => 1.0 / Math.Pow(theta, 2.0 * i / headDim)).ToArray();
        for (int h = 0; h < ph; h++)
            for (int w = 0; w < pw; w++)
            {
                var inv = freqs.Where((_, i) => i % 2 == 0).Select(f => h * f)
                    .Concat(freqs.Where((_, i) => i % 2 == 1).Select(f => w * f)).ToArray();
                var full = inv.Concat(inv).ToArray();
                for (int d = 0; d < headDim; d++)
                    Assert.Equal((float)full[d], got[(h * pw + w) * headDim + d], 5);
            }
    }

    /// <summary>The span the injector writes covers rows of [IMG] tokens, each followed by
    /// [IMG_BREAK] (the last by [IMG_END]); the embedding must line up with them.</summary>
    [Fact]
    public void ImageEmbedding_IsLaidOutRowByRow_WithTheMarkerTokenEmbeddings()
    {
        const int rows = 3, cols = 2, dim = 2;
        using var patches = new Tensor(_allocator, DType.Float32, rows * cols, dim);
        patches.SetElementsAsFloat(Enumerable.Range(0, rows * cols * dim).Select(i => (float)i).ToArray());
        using var markers = new Tensor(_allocator, DType.Float32, 2, dim);
        markers.SetElementsAsFloat(new[] { -1f, -1.5f, -2f, -2.5f }); // [IMG_BREAK], [IMG_END]

        using var laid = ModelMultimodalInjector.LayOutMistral3ImageRows(patches, markers, rows, cols);

        Assert.Equal(new long[] { rows * (cols + 1), dim }, laid.Sizes);
        Assert.Equal(new float[]
        {
            0, 1, 2, 3, -1, -1.5f,     // row 0 patches, [IMG_BREAK]
            4, 5, 6, 7, -1, -1.5f,     // row 1 patches, [IMG_BREAK]
            8, 9, 10, 11, -2, -2.5f,   // row 2 patches, [IMG_END]
        }, laid.GetElementsAsFloat((int)laid.ElementCount()));
    }

    [Fact]
    public void UnpermuteInterleavedRows_InvertsLlamaCppsPermute()
    {
        const int heads = 3, headDim = 6, cols = 5;
        float[] hf = Enumerable.Range(0, heads * headDim * cols).Select(i => (float)i).ToArray();
        Assert.Equal(hf, Mistral3VisionEncoder.UnpermuteInterleavedRows(PermuteLikeLlamaCpp(hf, heads, headDim), heads, headDim));
    }

    [Theory]
    [InlineData("v.pre_ln.weight")]
    [InlineData("mm.patch_merger.weight")]
    [InlineData("v.blk.0.ln2.weight")]
    public void ProjectorMissingATensor_IsRefusedAtLoad_NamingBothLayouts(string omitted)
    {
        string path = WriteProjector(Layout.LlamaCpp, "missing.gguf", n => n == omitted);
        var ex = Assert.Throws<NotSupportedException>(() => new Mistral3VisionEncoder(path, _allocator));
        Assert.Contains("missing", ex.Message);
        Assert.Contains("v.patch_embd", ex.Message);
        Assert.Contains("v.patch_conv", ex.Message);
    }

    [Fact]
    public void CanonicalTensorName_MapsLlamaCppClipNames_AndLeavesOthersAlone()
    {
        Assert.Equal("v.patch_conv.weight", Mistral3VisionEncoder.CanonicalTensorName("v.patch_embd.weight"));
        Assert.Equal("v.encoder_norm.weight", Mistral3VisionEncoder.CanonicalTensorName("v.pre_ln.weight"));
        Assert.Equal("v.blk.23.attn_norm.weight", Mistral3VisionEncoder.CanonicalTensorName("v.blk.23.ln1.weight"));
        Assert.Equal("v.blk.7.ffn_norm.weight", Mistral3VisionEncoder.CanonicalTensorName("v.blk.7.ln2.weight"));
        Assert.Equal("v.blk.0.attn_output.weight", Mistral3VisionEncoder.CanonicalTensorName("v.blk.0.attn_out.weight"));
        Assert.Equal("mm.norm.weight", Mistral3VisionEncoder.CanonicalTensorName("mm.input_norm.weight"));
        Assert.Equal("mm.patch_merger.merging_layer.weight", Mistral3VisionEncoder.CanonicalTensorName("mm.patch_merger.weight"));
        Assert.Equal("mm.linear_1.weight", Mistral3VisionEncoder.CanonicalTensorName("mm.1.weight"));
        Assert.Equal("mm.linear_2.weight", Mistral3VisionEncoder.CanonicalTensorName("mm.2.weight"));
        foreach (string same in new[] { "v.patch_conv.weight", "v.blk.0.attn_q.weight", "mm.linear_1.weight", "v.token_embd.img_break" })
            Assert.Equal(same, Mistral3VisionEncoder.CanonicalTensorName(same));
    }

    /// <summary>HF Mistral3PatchMerger: unfold over the [d, h, w] grid, so feature
    /// c*m*m + ky*m + kx of a merged token is channel c of window cell (ky, kx).</summary>
    [Fact]
    public unsafe void MergePatches_GathersEachWindowInTorchUnfoldChannelMajorOrder()
    {
        const int pw = 5, ph = 4, d = 3, m = 2; // odd width: the last column is dropped, as unfold does
        var src = new float[pw * ph * d];
        for (int y = 0; y < ph; y++)
            for (int x = 0; x < pw; x++)
                for (int c = 0; c < d; c++)
                    src[(y * pw + x) * d + c] = y * 100 + x * 10 + c;

        int mw = pw / m, mh = ph / m;
        var dst = new float[mw * mh * d * m * m];
        fixed (float* s = src)
        fixed (float* o = dst)
            Mistral3VisionEncoder.MergePatches(s, o, pw, ph, d, m);

        for (int my = 0; my < mh; my++)
            for (int mx = 0; mx < mw; mx++)
                for (int c = 0; c < d; c++)
                    for (int ky = 0; ky < m; ky++)
                        for (int kx = 0; kx < m; kx++)
                        {
                            float want = (my * m + ky) * 100 + (mx * m + kx) * 10 + c;
                            float got = dst[(my * mw + mx) * d * m * m + c * m * m + ky * m + kx];
                            Assert.Equal(want, got);
                        }
    }

    private static float[] Gen(string seed, int n, float scale, float offset)
    {
        uint state = 0x811c9dc5;
        foreach (byte ch in Encoding.UTF8.GetBytes(seed)) { state ^= ch; state = unchecked(state * 0x01000193); }
        if (state == 0) state = 1;
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 0xFFFF) / 32768f - 1f);
        }
        return data;
    }

    private static Action<BinaryWriter> U32(uint v) => w => { w.Write(4u); w.Write(v); };
    private static Action<BinaryWriter> F32(float v) => w => { w.Write(6u); w.Write(v); };
    private static Action<BinaryWriter> Bool(bool v) => w => { w.Write(7u); w.Write(v); };
    private static Action<BinaryWriter> Str(string v) => w => { w.Write(8u); WriteStr(w, v); };

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private const int Alignment = 32;

    private static void WriteGguf(Stream fs, List<(string Key, Action<BinaryWriter> Write)> kv,
        List<(string Name, ulong[] Dims, float[] Data)> tensors)
    {
        using var head = new MemoryStream();
        using (var w = new BinaryWriter(head, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);
            w.Write(3u);
            w.Write((ulong)tensors.Count);
            w.Write((ulong)kv.Count);
            foreach (var (key, write) in kv)
            {
                WriteStr(w, key);
                write(w);
            }
            ulong offset = 0;
            foreach (var (name, dims, data) in tensors)
            {
                WriteStr(w, name);
                w.Write((uint)dims.Length);
                foreach (ulong dim in dims) w.Write(dim);
                w.Write(0u); // F32
                w.Write(offset);
                ulong bytes = (ulong)data.Length * sizeof(float);
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        fs.Write(new byte[pad], 0, pad);
        foreach (var (_, _, data) in tensors)
        {
            byte[] raw = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            fs.Write(raw, 0, raw.Length);
            int tail = (Alignment - raw.Length % Alignment) % Alignment;
            fs.Write(new byte[tail], 0, tail);
        }
    }
}
