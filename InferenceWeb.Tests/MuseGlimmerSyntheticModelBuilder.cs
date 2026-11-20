// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Builds a tiny, deterministic Muse-Glimmer GGUF: four blocks by default (three sliding-window,
// one full NoPE), two KV heads, the attention output gate, the four per-layer norms
// and a softcapped LM head. Every 2D projection is F16, which the GGML backends store
// quantized-resident, so the model runs the whole-model fused kernel - and on a GPU
// backend its sliding-window layers get the ring KV cache - exactly as the 30B does,
// in milliseconds per forward. Weights come from a name-seeded generator, so two
// files written with the same options are bit-identical, and the default options write
// the bytes they always wrote.
//
// The options make the parts of the architecture a reimplementation can get wrong
// visible in the logits (the frozen legacy goldens, MuseGlimmerGoldens.cs): more blocks
// (two full layers among the sliding-window ones), a real logit_scale and a head large
// enough to reach the softcap's curve, a tied head, and small block-output projections,
// whose RMS is then close enough to the 1e-8 / 1e-5 epsilons for the post norms' epsilon
// to show.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerSyntheticModelBuilder
{
    public const int Hidden = 128;
    public const int NumHeads = 4;
    public const int NumKvHeads = 2;
    public const int HeadDim = 64;
    public const int FfnLength = 256;
    /// <summary>With the default sliding-window pattern of 4, blocks 0-2 are
    /// sliding-window layers and block 3 is a full-attention (NoPE) layer.</summary>
    public const int NumBlocks = 4;
    public const int ByteTokens = 256;

    public int SlidingWindow { get; init; } = 64;
    public int ContextLength { get; init; } = 8192;

    /// <summary>Decoder blocks; with the pattern period of 4, block l is a sliding-window layer when l % 4 &lt; 3.</summary>
    public int BlockCount { get; init; } = NumBlocks;

    /// <summary>muse-glimmer.logit_scale (the 30B ships 0.19612).</summary>
    public float LogitScale { get; init; } = 1f;

    /// <summary>muse-glimmer.final_logit_softcapping.</summary>
    public float Softcap { get; init; } = 20f;

    /// <summary>No output.weight: the head is token_embd.weight.</summary>
    public bool TiedOutput { get; init; }

    /// <summary>The range of token_embd.weight's values (uniform in [-scale, scale)).</summary>
    public float EmbeddingScale { get; init; } = 1.0f;

    /// <summary>The range of output.weight's values.</summary>
    public float HeadScale { get; init; } = 0.08f;

    /// <summary>The range of attn_output.weight's and ffn_down.weight's values: the block outputs the post norms take.</summary>
    public float BlockOutputScale { get; init; } = 0.08f;

    public int VocabSize => ByteTokens;

    public string Write(string path)
    {
        using var fs = File.Create(path);
        WriteGguf(fs, BuildMetadata(), BuildTensors());
        return path;
    }

    private enum Kind { F32, F16 }

    private List<(string Name, ulong[] Dims, float[] Data, Kind Type)> BuildTensors()
    {
        int qDim = NumHeads * HeadDim;
        int kvDim = NumKvHeads * HeadDim;
        var t = new List<(string, ulong[], float[], Kind)>
        {
            Gen("token_embd.weight", Kind.F16, EmbeddingScale, Hidden, VocabSize),
            Gen("output_norm.weight", Kind.F32, 0.2f, Hidden, offset: 1f),
        };
        if (!TiedOutput)
            t.Add(Gen("output.weight", Kind.F16, HeadScale, Hidden, VocabSize));
        for (int l = 0; l < BlockCount; l++)
        {
            string p = $"blk.{l}.";
            t.Add(Gen(p + "attn_norm.weight", Kind.F32, 0.2f, Hidden, offset: 1f));
            t.Add(Gen(p + "attn_q.weight", Kind.F16, 0.08f, Hidden, qDim));
            t.Add(Gen(p + "attn_k.weight", Kind.F16, 0.08f, Hidden, kvDim));
            t.Add(Gen(p + "attn_v.weight", Kind.F16, 0.08f, Hidden, kvDim));
            t.Add(Gen(p + "attn_gate.weight", Kind.F16, 0.08f, Hidden, qDim));
            t.Add(Gen(p + "attn_q_norm.weight", Kind.F32, 0.2f, HeadDim, offset: 1f));
            t.Add(Gen(p + "attn_k_norm.weight", Kind.F32, 0.2f, HeadDim, offset: 1f));
            t.Add(Gen(p + "attn_output.weight", Kind.F16, BlockOutputScale, qDim, Hidden));
            t.Add(Gen(p + "post_attention_norm.weight", Kind.F32, 0.2f, Hidden, offset: 1f));
            t.Add(Gen(p + "ffn_norm.weight", Kind.F32, 0.2f, Hidden, offset: 1f));
            t.Add(Gen(p + "ffn_gate.weight", Kind.F16, 0.08f, Hidden, FfnLength));
            t.Add(Gen(p + "ffn_up.weight", Kind.F16, 0.08f, Hidden, FfnLength));
            t.Add(Gen(p + "ffn_down.weight", Kind.F16, BlockOutputScale, FfnLength, Hidden));
            t.Add(Gen(p + "post_ffw_norm.weight", Kind.F32, 0.2f, Hidden, offset: 1f));
        }
        return t;
    }

    private static (string, ulong[], float[], Kind) Gen(string name, Kind kind, float scale, int d0, int d1 = 0, float offset = 0f)
    {
        int n = d1 > 0 ? d0 * d1 : d0;
        uint state = Fnv1a(name);
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 0xFFFF) / 32768f - 1f);
        }
        return (name, d1 > 0 ? new[] { (ulong)d0, (ulong)d1 } : new[] { (ulong)d0 }, data, kind);
    }

    private static uint Fnv1a(string text)
    {
        uint h = 0x811c9dc5;
        foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h = unchecked(h * 0x01000193); }
        return h == 0 ? 1u : h;
    }

    private List<(string Key, Action<BinaryWriter> Write)> BuildMetadata()
    {
        const string a = "muse-glimmer";
        var kv = new List<(string, Action<BinaryWriter>)>
        {
            ("general.architecture", Str(a)),
            ($"{a}.block_count", U32((uint)BlockCount)),
            ($"{a}.context_length", U32((uint)ContextLength)),
            ($"{a}.embedding_length", U32(Hidden)),
            ($"{a}.feed_forward_length", U32(FfnLength)),
            ($"{a}.attention.head_count", U32(NumHeads)),
            ($"{a}.attention.head_count_kv", U32(NumKvHeads)),
            ($"{a}.attention.key_length", U32(HeadDim)),
            ($"{a}.attention.value_length", U32(HeadDim)),
            ($"{a}.rope.freq_base", F32(500000f)),
            ($"{a}.attention.layer_norm_rms_epsilon", F32(1e-5f)),
            ($"{a}.final_logit_softcapping", F32(Softcap)),
            ($"{a}.logit_scale", F32(LogitScale)),
            ($"{a}.attention.sliding_window", U32((uint)SlidingWindow)),
            ($"{a}.attention.sliding_window_pattern", U32(4)),
        };

        var tokens = new List<string>(VocabSize);
        foreach (int cp in ByteToUnicode()) tokens.Add(char.ConvertFromUtf32(cp));
        var types = new List<int>();
        for (int i = 0; i < ByteTokens; i++) types.Add(1);

        kv.Add(("tokenizer.ggml.model", Str("gpt2")));
        kv.Add(("tokenizer.ggml.pre", Str("llama4")));
        kv.Add(("tokenizer.ggml.tokens", StrArr(tokens)));
        kv.Add(("tokenizer.ggml.token_type", I32Arr(types)));
        kv.Add(("tokenizer.ggml.merges", StrArr(Array.Empty<string>())));
        kv.Add(("tokenizer.ggml.bos_token_id", U32('A')));
        kv.Add(("tokenizer.ggml.eos_token_id", U32(0)));
        kv.Add(("tokenizer.ggml.add_bos_token", Bool(false)));
        kv.Add(("tokenizer.ggml.add_eos_token", Bool(false)));
        return kv;
    }

    private static Action<BinaryWriter> U32(uint v) => w => { w.Write(4u); w.Write(v); };
    private static Action<BinaryWriter> F32(float v) => w => { w.Write(6u); w.Write(v); };
    private static Action<BinaryWriter> Bool(bool v) => w => { w.Write(7u); w.Write(v); };
    private static Action<BinaryWriter> Str(string v) => w => { w.Write(8u); WriteStr(w, v); };
    private static Action<BinaryWriter> StrArr(IReadOnlyList<string> v) => w =>
    {
        w.Write(9u); w.Write(8u); w.Write((ulong)v.Count);
        foreach (string s in v) WriteStr(w, s);
    };
    private static Action<BinaryWriter> I32Arr(IReadOnlyList<int> v) => w =>
    {
        w.Write(9u); w.Write(5u); w.Write((ulong)v.Count);
        foreach (int x in v) w.Write(x);
    };

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    private static IEnumerable<int> ByteToUnicode()
    {
        var bs = new List<int>();
        for (int b = '!'; b <= '~'; b++) bs.Add(b);
        for (int b = 0xA1; b < 0xAD; b++) bs.Add(b);
        for (int b = 0xAE; b < 0x100; b++) bs.Add(b);
        var cs = new List<int>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (bs.Contains(b)) continue;
            bs.Add(b);
            cs.Add(256 + n);
            n++;
        }
        var map = new int[256];
        for (int i = 0; i < bs.Count; i++) map[bs[i]] = cs[i];
        return map;
    }

    private const int Alignment = 32;

    private static byte[] Encode(float[] data, Kind kind)
    {
        if (kind == Kind.F32)
        {
            byte[] raw = new byte[data.Length * sizeof(float)];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            return raw;
        }
        byte[] half = new byte[data.Length * 2];
        for (int i = 0; i < data.Length; i++)
            BitConverter.TryWriteBytes(half.AsSpan(i * 2, 2), (System.Half)data[i]);
        return half;
    }

    private static void WriteGguf(Stream fs, List<(string Key, Action<BinaryWriter> Write)> kv,
        List<(string Name, ulong[] Dims, float[] Data, Kind Type)> tensors)
    {
        var payloads = new List<byte[]>(tensors.Count);
        foreach (var t in tensors)
            payloads.Add(Encode(t.Data, t.Type));

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
            for (int i = 0; i < tensors.Count; i++)
            {
                var (name, dims, _, kind) = tensors[i];
                WriteStr(w, name);
                w.Write((uint)dims.Length);
                foreach (ulong d in dims) w.Write(d);
                w.Write(kind == Kind.F32 ? 0u : 1u);   // GGML_TYPE_F32 / GGML_TYPE_F16
                w.Write(offset);
                ulong bytes = (ulong)payloads[i].Length;
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        fs.Write(new byte[pad], 0, pad);
        foreach (byte[] raw in payloads)
        {
            fs.Write(raw, 0, raw.Length);
            int tail = (Alignment - raw.Length % Alignment) % Alignment;
            fs.Write(new byte[tail], 0, tail);
        }
    }
}
