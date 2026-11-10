// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Builds a tiny, deterministic dense-decoder GGUF (llama-style block: GQA, SwiGLU,
// RMSNorm, optional per-head Q/K norms) so architecture ROUTING and the KV-state
// snapshot contract can be tested without a multi-GB checkpoint. Weights are F32
// from a name-seeded generator, so two files written with the same options but a
// different label are bit-identical tensors under different metadata.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace InferenceWeb.Tests;

public sealed class DenseDecoderSyntheticModelBuilder
{
    public const int Hidden = 64;
    public const int NumHeads = 4;
    public const int NumKvHeads = 2;
    public const int HeadDim = 16;
    public const int FfnLength = 128;
    public const int NumBlocks = 2;
    public const int ByteTokens = 256;

    /// <summary>general.architecture, and the prefix of every hyperparameter key.</summary>
    public string Architecture { get; init; } = "llama";
    public string MetadataPrefix { get; init; }
    public string PreTokenizer { get; init; } = "tekken";
    public bool IncludeMistralControlTokens { get; init; } = true;
    public bool IncludeRopeFreqs { get; init; }
    public bool IncludeQkNorms { get; init; }
    public uint ExpertCount { get; init; }

    /// <summary>Routed experts per layer (0: dense FFN). Each layer then carries a router
    /// (<c>ffn_gate_inp</c>) and stacked F16 <c>ffn_{gate,up,down}_exps</c>, which load as the
    /// stacked quantized experts a real MoE checkpoint gives, plus the expert metadata.</summary>
    public int MoeExperts { get; init; }
    public int MoeTopK { get; init; } = 2;
    /// <summary>Add an always-active shared expert (<c>ffn_*_shexp</c>) to every MoE layer.</summary>
    public bool MoeSharedExpert { get; init; }
    public const int ExpertFfnLength = 32;
    public string RopeScalingType { get; init; }
    public float RopeBase { get; init; } = 1_000_000f;
    public int LayerCount { get; init; } = NumBlocks;
    /// <summary>Key/value head dimension (default <see cref="HeadDim"/>; flash-attention kernels need 64 and up).</summary>
    public int KeyLength { get; init; } = HeadDim;
    public bool TiedOutput { get; init; }
    /// <summary>Store one layer's matrices as F16 while all other weights remain F32.</summary>
    public int? HalfPrecisionLayer { get; init; }

    /// <summary>
    /// Rewrite tensors before they are written (a LoRA test's merged-weights twin): called
    /// with each GGUF tensor name and its values as the file would store them (an F16 layer's
    /// already rounded to half), it returns the values to write, or null to keep them. A
    /// rewritten tensor is stored as F32, so the twin holds exactly what it was given.
    /// </summary>
    public Func<string, float[], float[]> Transform { get; init; }

    private static readonly string[] MistralControlTokens = { "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]" };

    public int VocabSize => ByteTokens + (IncludeMistralControlTokens ? MistralControlTokens.Length : 0);

    public string Write(string path)
    {
        var tensors = BuildTensors();
        var forcedF32 = new HashSet<string>(StringComparer.Ordinal);
        if (Transform != null)
        {
            for (int i = 0; i < tensors.Count; i++)
            {
                var (name, dims, data) = tensors[i];
                float[] stored = IsHalf(name, dims) ? Array.ConvertAll(data, v => (float)(Half)v) : data;
                float[] rewritten = Transform(name, stored);
                if (rewritten == null) continue;
                tensors[i] = (name, dims, rewritten);
                forcedF32.Add(name);
            }
        }
        using var fs = File.Create(path);
        WriteGguf(fs, BuildMetadata(), tensors, forcedF32);
        return path;
    }

    private bool IsHalf(string name, ulong[] dims) =>
        (dims.Length == 3 && name.Contains("_exps.", StringComparison.Ordinal)) ||
        (HalfPrecisionLayer.HasValue && dims.Length == 2 &&
         name.StartsWith($"blk.{HalfPrecisionLayer.Value}.", StringComparison.Ordinal));

    private List<(string Name, ulong[] Dims, float[] Data)> BuildTensors()
    {
        int qDim = NumHeads * KeyLength;
        int kDim = NumKvHeads * KeyLength;
        var t = new List<(string, ulong[], float[])>
        {
            Gen("token_embd.weight", 0.08f, Hidden, VocabSize),
            Gen("output_norm.weight", 0.5f, Hidden, offset: 1f),
        };
        if (!TiedOutput) t.Add(Gen("output.weight", 0.08f, Hidden, VocabSize));
        if (IncludeRopeFreqs)
            t.Add(Gen("rope_freqs.weight", 0.1f, KeyLength / 2, offset: 1f));

        for (int l = 0; l < LayerCount; l++)
        {
            string p = $"blk.{l}.";
            t.Add(Gen(p + "attn_norm.weight", 0.2f, Hidden, offset: 1f));
            t.Add(Gen(p + "attn_q.weight", 0.08f, Hidden, qDim));
            t.Add(Gen(p + "attn_k.weight", 0.08f, Hidden, kDim));
            t.Add(Gen(p + "attn_v.weight", 0.08f, Hidden, kDim));
            t.Add(Gen(p + "attn_output.weight", 0.08f, qDim, Hidden));
            if (IncludeQkNorms)
            {
                t.Add(Gen(p + "attn_q_norm.weight", 0.2f, KeyLength, offset: 1f));
                t.Add(Gen(p + "attn_k_norm.weight", 0.2f, KeyLength, offset: 1f));
            }
            t.Add(Gen(p + "ffn_norm.weight", 0.2f, Hidden, offset: 1f));
            if (MoeExperts > 0)
            {
                t.Add(Gen(p + "ffn_gate_inp.weight", 0.3f, Hidden, MoeExperts));
                t.Add(Gen3(p + "ffn_gate_exps.weight", 0.1f, Hidden, ExpertFfnLength, MoeExperts));
                t.Add(Gen3(p + "ffn_up_exps.weight", 0.1f, Hidden, ExpertFfnLength, MoeExperts));
                t.Add(Gen3(p + "ffn_down_exps.weight", 0.1f, ExpertFfnLength, Hidden, MoeExperts));
                if (MoeSharedExpert)
                {
                    t.Add(Gen(p + "ffn_gate_shexp.weight", 0.08f, Hidden, ExpertFfnLength));
                    t.Add(Gen(p + "ffn_up_shexp.weight", 0.08f, Hidden, ExpertFfnLength));
                    t.Add(Gen(p + "ffn_down_shexp.weight", 0.08f, ExpertFfnLength, Hidden));
                }
                continue;
            }
            t.Add(Gen(p + "ffn_gate.weight", 0.08f, Hidden, FfnLength));
            t.Add(Gen(p + "ffn_up.weight", 0.08f, Hidden, FfnLength));
            t.Add(Gen(p + "ffn_down.weight", 0.08f, FfnLength, Hidden));
        }
        return t;
    }

    private static (string, ulong[], float[]) Gen(string name, float scale, int d0, int d1 = 0, float offset = 0f)
    {
        int n = d1 > 0 ? d0 * d1 : d0;
        uint state = Fnv1a(name);
        var data = new float[n];
        for (int i = 0; i < n; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            data[i] = offset + scale * ((state & 0xFFFF) / 32768f - 1f);
        }
        return (name, d1 > 0 ? new[] { (ulong)d0, (ulong)d1 } : new[] { (ulong)d0 }, data);
    }

    private static (string, ulong[], float[]) Gen3(string name, float scale, int d0, int d1, int d2)
    {
        var (_, _, data) = Gen(name, scale, d0 * d1, d2);
        return (name, new[] { (ulong)d0, (ulong)d1, (ulong)d2 }, data);
    }

    private static uint Fnv1a(string text)
    {
        uint h = 0x811c9dc5;
        foreach (byte b in Encoding.UTF8.GetBytes(text)) { h ^= b; h = unchecked(h * 0x01000193); }
        return h == 0 ? 1u : h;
    }

    private List<(string Key, Action<BinaryWriter> Write)> BuildMetadata()
    {
        string a = MetadataPrefix ?? Architecture;
        var kv = new List<(string, Action<BinaryWriter>)>
        {
            ("general.architecture", Str(Architecture)),
            ($"{a}.block_count", U32(checked((uint)LayerCount))),
            ($"{a}.context_length", U32(4096)),
            ($"{a}.embedding_length", U32(Hidden)),
            ($"{a}.feed_forward_length", U32(FfnLength)),
            ($"{a}.attention.head_count", U32(NumHeads)),
            ($"{a}.attention.head_count_kv", U32(NumKvHeads)),
            ($"{a}.attention.key_length", U32((uint)KeyLength)),
            ($"{a}.attention.value_length", U32((uint)KeyLength)),
            ($"{a}.rope.dimension_count", U32((uint)KeyLength)),
            ($"{a}.rope.freq_base", F32(RopeBase)),
            ($"{a}.attention.layer_norm_rms_epsilon", F32(1e-5f)),
            ($"{a}.vocab_size", U32((uint)VocabSize)),
        };
        if (ExpertCount > 0)
            kv.Add(($"{a}.expert_count", U32(ExpertCount)));
        if (MoeExperts > 0)
        {
            kv.Add(($"{a}.expert_count", U32((uint)MoeExperts)));
            kv.Add(($"{a}.expert_used_count", U32((uint)MoeTopK)));
            kv.Add(($"{a}.expert_feed_forward_length", U32(ExpertFfnLength)));
        }
        if (RopeScalingType != null)
            kv.Add(($"{a}.rope.scaling.type", Str(RopeScalingType)));

        var tokens = new List<string>(VocabSize);
        foreach (int cp in ByteToUnicode()) tokens.Add(char.ConvertFromUtf32(cp));
        var types = new List<int>();
        for (int i = 0; i < ByteTokens; i++) types.Add(1);
        if (IncludeMistralControlTokens)
        {
            foreach (string c in MistralControlTokens) { tokens.Add(c); types.Add(3); }
        }

        kv.Add(("tokenizer.ggml.model", Str("gpt2")));
        kv.Add(("tokenizer.ggml.pre", Str(PreTokenizer)));
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

    private void WriteGguf(Stream fs, List<(string Key, Action<BinaryWriter> Write)> kv,
        List<(string Name, ulong[] Dims, float[] Data)> tensors, HashSet<string> forcedF32)
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
                foreach (ulong d in dims) w.Write(d);
                bool half = IsHalf(name, dims) && !forcedF32.Contains(name);
                w.Write(half ? 1u : 0u); // F16 or F32
                w.Write(offset);
                ulong bytes = (ulong)data.Length * (ulong)(half ? sizeof(ushort) : sizeof(float));
                offset += (bytes + Alignment - 1) / Alignment * Alignment;
            }
        }

        byte[] headBytes = head.ToArray();
        fs.Write(headBytes, 0, headBytes.Length);
        int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
        fs.Write(new byte[pad], 0, pad);
        foreach (var (name, dims, data) in tensors)
        {
            bool half = IsHalf(name, dims) && !forcedF32.Contains(name);
            byte[] raw = new byte[data.Length * (half ? sizeof(ushort) : sizeof(float))];
            if (half)
                for (int i = 0; i < data.Length; i++)
                    System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(i * sizeof(ushort)),
                        BitConverter.HalfToUInt16Bits((Half)data[i]));
            else Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            fs.Write(raw, 0, raw.Length);
            int tail = (Alignment - raw.Length % Alignment) % Alignment;
            fs.Write(new byte[tail], 0, tail);
        }
    }
}
