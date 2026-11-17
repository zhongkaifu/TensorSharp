// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using static InferenceWeb.Tests.SyntheticGguf;

namespace InferenceWeb.Tests;

/// <summary>
/// A small LiquidAI LFM3 checkpoint (GGUF <c>lfm3</c>) with every mechanism of the real one: short-conv layers
/// (head_count_kv 0, <c>shortconv.l_cache</c> 7), full and sliding-window attention layers (a window of 8, so a 17-token
/// prompt already slides), per-head Q/K norms, the head-wise output gate, sandwich norms, the zero-tail
/// <c>rope_freqs.weight</c> of the full layers (half the pairs rotate) and the two rotary bases, a tied head by default.
/// </summary>
internal sealed class Lfm3SyntheticModelBuilder
{
    public const int Hidden = 256;
    public const int Heads = 4;
    public const int KvHeads = 2;
    public const int HeadDim = 64;
    public const int Ff = 512;
    public const int Vocab = 256;
    public const int Window = 8;
    public const int ConvTaps = 7;
    public const float RopeBase = 1000000f;
    public const float RopeBaseSliding = 10000f;
    public const float Eps = 1e-5f;

    /// <summary>Layer kinds: 'c' short conv, 'f' full attention, 's' sliding-window attention.</summary>
    public string Layout { get; init; } = "csfcsfsc";
    /// <summary>The projections' type (norms, the conv filter and rope_freqs stay F32).</summary>
    public GgmlType WeightType { get; init; } = GgmlType.BF16;
    /// <summary>The FFN projections as NVFP4 with per-tensor .scale (and unused .input_scale) sidecars, as LiquidAI's NVFP4
    /// checkpoint ships them.</summary>
    public bool Nvfp4Ffn { get; init; }
    /// <summary>An untied output.weight (the real checkpoints tie the head to token_embd).</summary>
    public bool UntiedOutput { get; init; }
    /// <summary>Rotated pairs of the full layers (rope_freqs ones; the rest 1e30).</summary>
    public int RotatedPairs { get; init; } = HeadDim / 4;

    public int Layers => Layout.Length;
    public bool IsConv(int l) => Layout[l] == 'c';
    public bool IsSliding(int l) => Layout[l] == 's';

    /// <summary>Every tensor with its F32 values (what <see cref="Write"/> stores, before any quantization).</summary>
    public List<Tensor> Tensors()
    {
        var t = new List<Tensor>
        {
            Gen("token_embd.weight", 0.5f, Hidden, Vocab),
            GenAround("output_norm.weight", 1f, 0.1f, Hidden),
        };
        var factors = new float[HeadDim / 2];
        for (int i = 0; i < factors.Length; i++) factors[i] = i < RotatedPairs ? 1f : 1.0000000150474662e+30f;
        t.Add(new Tensor { Name = "rope_freqs.weight", Dims = new ulong[] { (ulong)factors.Length }, Data = factors });
        if (UntiedOutput) t.Add(Gen("output.weight", 0.08f, Hidden, Vocab));
        for (int l = 0; l < Layers; l++)
        {
            string p = $"blk.{l}.";
            t.Add(GenAround(p + "attn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "post_attention_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "ffn_norm.weight", 1f, 0.1f, Hidden));
            t.Add(GenAround(p + "post_ffw_norm.weight", 1f, 0.1f, Hidden));
            if (IsConv(l))
            {
                t.Add(Gen(p + "shortconv.in_proj.weight", 0.08f, Hidden, 3 * Hidden));
                t.Add(Gen(p + "shortconv.conv.weight", 0.3f, ConvTaps, Hidden));
                t.Add(Gen(p + "shortconv.out_proj.weight", 0.06f, Hidden, Hidden));
            }
            else
            {
                t.Add(Gen(p + "attn_q.weight", 0.06f, Hidden, Heads * HeadDim));
                t.Add(Gen(p + "attn_k.weight", 0.06f, Hidden, KvHeads * HeadDim));
                t.Add(Gen(p + "attn_v.weight", 0.06f, Hidden, KvHeads * HeadDim));
                t.Add(GenAround(p + "attn_q_norm.weight", 1f, 0.2f, HeadDim));
                t.Add(GenAround(p + "attn_k_norm.weight", 1f, 0.2f, HeadDim));
                t.Add(Gen(p + "attn_gate.weight", 0.15f, Hidden, Heads));
                t.Add(Gen(p + "attn_output.weight", 0.05f, Heads * HeadDim, Hidden));
            }
            t.Add(Gen(p + "ffn_gate.weight", 0.06f, Hidden, Ff));
            t.Add(Gen(p + "ffn_up.weight", 0.06f, Hidden, Ff));
            t.Add(Gen(p + "ffn_down.weight", 0.05f, Ff, Hidden));
        }
        return t;
    }

    public string Write(string path)
    {
        const string a = "lfm3";
        var kvPerLayer = new int[Layers];
        var pattern = new bool[Layers];
        for (int l = 0; l < Layers; l++)
        {
            kvPerLayer[l] = IsConv(l) ? 0 : KvHeads;
            pattern[l] = IsSliding(l);
        }
        var kv = new List<Kv>
        {
            new Str { Key = "general.architecture", V = a },
            new U32 { Key = $"{a}.block_count", V = (uint)Layers },
            new U32 { Key = $"{a}.context_length", V = 4096 },
            new U32 { Key = $"{a}.embedding_length", V = Hidden },
            new U32 { Key = $"{a}.feed_forward_length", V = Ff },
            new U32 { Key = $"{a}.attention.head_count", V = Heads },
            new I32Arr { Key = $"{a}.attention.head_count_kv", V = kvPerLayer },
            new U32 { Key = $"{a}.attention.key_length", V = HeadDim },
            new U32 { Key = $"{a}.attention.value_length", V = HeadDim },
            new U32 { Key = $"{a}.rope.dimension_count", V = HeadDim },
            new F32 { Key = $"{a}.rope.freq_base", V = RopeBase },
            new F32 { Key = $"{a}.rope.freq_base_swa", V = RopeBaseSliding },
            new U32 { Key = $"{a}.attention.sliding_window", V = Window },
            new BoolArr { Key = $"{a}.attention.sliding_window_pattern", V = pattern },
            new U32 { Key = $"{a}.shortconv.l_cache", V = ConvTaps },
            new F32 { Key = $"{a}.attention.layer_norm_rms_epsilon", V = Eps },
        };
        AddByteTokenizer(kv, Vocab, bos: 1, eos: 2);

        var tensors = new List<Tensor>();
        foreach (Tensor x in Tensors())
        {
            bool coefficient = x.Dims.Length == 1 || x.Name.EndsWith(".shortconv.conv.weight", StringComparison.Ordinal);
            bool ffn = x.Name.Contains(".ffn_", StringComparison.Ordinal) && !x.Name.EndsWith("_norm.weight", StringComparison.Ordinal);
            if (coefficient) { tensors.Add(x); continue; }
            if (Nvfp4Ffn && ffn)
            {
                // A per-tensor scale2 far from 1 (as LiquidAI's, ~2e-4): forgetting it is never close.
                float scale2 = 2.5e-3f;
                tensors.Add(Nvfp4(x, scale2));
                string stem = x.Name[..^".weight".Length];
                tensors.Add(new Tensor { Name = stem + ".scale", Dims = new ulong[] { 1 }, Data = new[] { scale2 } });
                tensors.Add(new Tensor { Name = stem + ".input_scale", Dims = new ulong[] { 1 }, Data = new[] { 0.125f } });
                continue;
            }
            x.Type = WeightType;
            tensors.Add(x);
        }
        SyntheticGguf.Write(path, kv, tensors);
        return path;
    }

    /// <summary>
    /// <paramref name="x"/> divided by <paramref name="scale2"/> as NVFP4 blocks (64 values: four UE4M3 sub-block scales, then
    /// 32 bytes of E2M1 codes; ggml's dequantize_row_nvfp4 reads value = kvalue[code] * ue4m3(d) / 2), so the weight times its
    /// .scale sidecar approximates <paramref name="x"/>.
    /// </summary>
    public static Tensor Nvfp4(Tensor x, float scale2)
    {
        const int block = 64, sub = 16, bytes = 36;
        int[] kvalues = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
        if (x.Data.Length % block != 0) throw new ArgumentException($"NVFP4 rows need a multiple of {block} values: '{x.Name}'.");
        var raw = new byte[x.Data.Length / block * bytes];
        for (int b = 0; b < x.Data.Length / block; b++)
            for (int s = 0; s < block / sub; s++)
            {
                float max = 0;
                for (int j = 0; j < sub; j++) max = Math.Max(max, Math.Abs(x.Data[b * block + s * sub + j] / scale2));
                byte d = Ue4m3(max / 12f * 2f);   // the largest code (12) times d / 2 reaches the sub-block's largest value
                raw[b * bytes + s] = d;
                float scale = Ue4m3ToFloat(d) * 0.5f;
                for (int j = 0; j < sub; j++)
                {
                    float v = scale == 0 ? 0 : x.Data[b * block + s * sub + j] / scale2 / scale;
                    int best = 0;
                    for (int c = 1; c < 16; c++)
                        if (Math.Abs(kvalues[c] - v) < Math.Abs(kvalues[best] - v)) best = c;
                    // Value j of the sub-block is the low nibble of byte j (j < 8) or the high nibble of byte j - 8.
                    int at = b * bytes + 4 + s * (sub / 2) + j % (sub / 2);
                    raw[at] |= (byte)(j < sub / 2 ? best : best << 4);
                }
            }
        return new Tensor { Name = x.Name, Dims = x.Dims, Type = (GgmlType)40, RawBlocks = raw };
    }

    /// <summary>ggml_ue4m3_to_fp32 without its final halving: (1 + m / 8) * 2^(e - 7), or m * 2^-9 at e = 0.</summary>
    public static float Ue4m3ToFloat(byte x)
    {
        if (x == 0 || x == 0x7F) return 0f;
        int e = (x >> 3) & 0xF, m = x & 7;
        return e == 0 ? MathF.ScaleB(m, -9) : MathF.ScaleB(1f + m / 8f, e - 7);
    }

    /// <summary>The UE4M3 byte nearest above-or-at <paramref name="v"/> (0x7F, a NaN that decodes as 0, is never chosen).</summary>
    private static byte Ue4m3(float v)
    {
        byte best = 1;
        for (int b = 1; b < 0x7F; b++)
            if (Ue4m3ToFloat((byte)b) >= v) { best = (byte)b; break; }
            else best = (byte)b;
        return best;
    }
}
