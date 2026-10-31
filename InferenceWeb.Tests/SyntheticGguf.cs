// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace InferenceWeb.Tests;

/// <summary>
/// Writing small synthetic GGUF checkpoints for engine tests: deterministic weights (a tensor's
/// values depend only on its name), the tensor types the engines load (F32, F16, BF16, Q8_0
/// quantized exactly as ggml's quantize_row_q8_0 does, and the K/IQ block formats as seeded block
/// bytes), the metadata value kinds architectures read, and GPT-2's byte-level tokenizer over the
/// first <c>vocab</c> bytes.
/// </summary>
internal static class SyntheticGguf
{
    public enum GgmlType { F32 = 0, F16 = 1, Q4_0 = 2, Q5_1 = 7, Q8_0 = 8, Q4_K = 12, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23, BF16 = 30, MXFP4 = 39 }

    private const int Q8Block = 32;
    private const int Q8BlockBytes = 2 + Q8Block;   // f16 scale + 32 int8
    private const int Alignment = 32;

    /// <summary>Values per block and bytes per block of a block format.</summary>
    private static (int Values, int Bytes) BlockOf(GgmlType type) => type switch
    {
        GgmlType.Q8_0 => (32, 34),
        GgmlType.Q4_0 => (32, 18),      // f16 d, 16 bytes of 4-bit codes (d * (code - 8))
        GgmlType.Q5_1 => (32, 24),      // f16 d, f16 m, high bits[4], low bits[16]
        GgmlType.Q4_K => (256, 144),    // f16 d, f16 dmin, scales[12], quant bits[128]
        GgmlType.IQ4_NL => (32, 18),     // f16 d, 16 bytes of 4-bit indices
        GgmlType.Q6_K => (256, 210),     // ql[128], qh[64], int8 scales[16], f16 d
        GgmlType.IQ3_S => (256, 110),    // f16 d, qs[64], qh[8], signs[32], scales[4]
        GgmlType.IQ4_XS => (256, 136),   // f16 d, u16 scales_h, scales_l[4], qs[128]
        GgmlType.IQ2_XS => (256, 74),    // f16 d, qs[32] u16 (9-bit grid + 7-bit signs), scales[8]
        GgmlType.IQ3_XXS => (256, 98),   // f16 d, qs[64] grid bytes, 8 u32 of signs + scale
        GgmlType.IQ2_XXS => (256, 66),   // f16 d, qs[32] u16 (grid bytes, then signs + scale)
        GgmlType.IQ2_S => (256, 82),     // f16 d, qs[64] (32 grid bytes + 32 sign bytes), qh[8], scales[8]
        GgmlType.IQ1_S => (256, 50),     // f16 d, qs[32] grid bytes, u16 qh[8] (grid high bits, scale, delta sign)
        GgmlType.MXFP4 => (32, 17),      // u8 E8M0 exponent e, 16 bytes of 4-bit E2M1 codes
        _ => (1, 0),
    };

    public sealed class Tensor
    {
        public string Name;
        public ulong[] Dims;     // GGUF order (ne0 fastest)
        public float[] Data;
        public GgmlType Type = GgmlType.F32;
        /// <summary>A block format written as seeded block bytes (<see cref="Blocks"/>) rather than
        /// quantized from <see cref="Data"/>: every byte pattern of these formats decodes to finite
        /// values, and the engines under test read the same bytes.</summary>
        public byte[] RawBlocks;

        public long ElementCount
        {
            get
            {
                long n = 1;
                foreach (ulong d in Dims) n *= (long)d;
                return n;
            }
        }

        public ulong ByteCount => RawBlocks != null ? (ulong)RawBlocks.Length : Type switch
        {
            GgmlType.Q8_0 => (ulong)Data.Length / Q8Block * Q8BlockBytes,
            GgmlType.F16 or GgmlType.BF16 => (ulong)Data.Length * 2,
            _ => (ulong)Data.Length * sizeof(float),
        };

        public byte[] Raw()
        {
            if (RawBlocks != null)
                return RawBlocks;
            switch (Type)
            {
                case GgmlType.Q8_0:
                    return QuantizeQ8_0(Data, (int)Dims[0]);
                case GgmlType.F16:
                {
                    var raw = new byte[Data.Length * 2];
                    for (int i = 0; i < Data.Length; i++)
                        BitConverter.TryWriteBytes(raw.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((System.Half)Data[i]));
                    return raw;
                }
                case GgmlType.BF16:
                {
                    // Round to nearest even on the upper 16 bits, as ggml's fp32->bf16 does.
                    var raw = new byte[Data.Length * 2];
                    for (int i = 0; i < Data.Length; i++)
                    {
                        uint u = BitConverter.SingleToUInt32Bits(Data[i]);
                        u += 0x7FFF + ((u >> 16) & 1);
                        BitConverter.TryWriteBytes(raw.AsSpan(i * 2), (ushort)(u >> 16));
                    }
                    return raw;
                }
                default:
                {
                    var raw = new byte[Data.Length * sizeof(float)];
                    Buffer.BlockCopy(Data, 0, raw, 0, raw.Length);
                    return raw;
                }
            }
        }
    }

    /// <summary>A tensor of roughly normal values of the given scale, seeded by its name.</summary>
    public static Tensor Gen(string name, float scale, params int[] dims) => GenAround(name, 0f, scale, dims);

    /// <summary><see cref="Gen"/> shifted by <paramref name="center"/> (norm weights near 1, decays
    /// below 0).</summary>
    public static Tensor GenAround(string name, float center, float scale, params int[] dims)
    {
        long n = 1;
        foreach (int d in dims) n *= d;
        var rng = new Rng(Fnv1a(name));
        var data = new float[n];
        for (long i = 0; i < n; i++) data[i] = center + rng.Normalish(scale);
        var ul = new ulong[dims.Length];
        for (int i = 0; i < dims.Length; i++) ul[i] = (ulong)dims[i];
        return new Tensor { Name = name, Dims = ul, Data = data };
    }

    /// <summary>
    /// A tensor of a K/IQ block format as seeded block bytes: random quant bits, the block scale an
    /// F16 near <paramref name="scale"/> divided by the format's typical code magnitude, so decoded
    /// weights come out roughly <paramref name="scale"/> in size.
    /// </summary>
    public static Tensor Blocks(string name, GgmlType type, float scale, params int[] dims)
    {
        var (valuesPerBlock, bytesPerBlock) = BlockOf(type);
        if (bytesPerBlock == 0)
            throw new ArgumentException($"{type} is not a block format", nameof(type));
        if (dims[0] % valuesPerBlock != 0)
            throw new ArgumentException($"{name}: row length {dims[0]} is not a multiple of {type}'s {valuesPerBlock}");
        long n = 1;
        foreach (int d in dims) n *= d;
        long blocks = n / valuesPerBlock;
        var raw = new byte[blocks * bytesPerBlock];
        var rng = new Rng(Fnv1a(name));
        for (long i = 0; i < raw.Length; i++) raw[i] = (byte)(rng.Next() >> 24);
        // The typical |decoded code| of each format, which sets the block scale d.
        float typical = type switch
        {
            GgmlType.Q5_1 => 16f,          // unsigned 5-bit code with a negative minimum
            GgmlType.Q4_0 => 8f,           // 4-bit code minus 8
            GgmlType.Q4_K => 16f * 32f,    // unsigned 4-bit code and 6-bit sub-scales
            GgmlType.IQ4_NL => 64f,          // kvalues_iq4nl span [-127, 113]
            GgmlType.IQ4_XS => 64f * 16f,    // times a 6-bit sub-scale minus 32
            GgmlType.IQ3_S => 16f * 7f,      // (1 + 2 * 4-bit scale) times grid values up to 15
            GgmlType.Q6_K => 64f * 16f,      // int8 scale times a 6-bit code minus 32
            GgmlType.IQ2_XS => 25f * 2f,     // grid values 8/25/43 times (0.5 + 4-bit scale) / 4
            GgmlType.IQ3_XXS => 28f * 4f,    // grid values 4..62 times (0.5 + 4-bit scale) / 2
            GgmlType.IQ2_XXS or GgmlType.IQ2_S => 25f * 2f, // grid values 8/25/43 times (0.5 + 4-bit scale) / 4
            GgmlType.IQ1_S => 8f,          // (2 * 3-bit scale + 1) times grid values -1..1 +/- 1/8
            _ => 1f,
        };
        for (long b = 0; b < blocks; b++)
        {
            if (type == GgmlType.MXFP4)
            {
                // value = kvalues_mxfp4[code] (|code| averages 4.5, at most 12) * 2^(e - 128),
                // ggml's GGML_E8M0_TO_FP32_HALF; e jitters by one either side of the target.
                int e = 128 + (int)MathF.Round(MathF.Log2(scale / 4.5f)) + (int)(rng.Next() % 3) - 1;
                raw[b * bytesPerBlock] = (byte)Math.Clamp(e, 1, 254);
                continue;
            }
            long at = b * bytesPerBlock + (type == GgmlType.Q6_K ? bytesPerBlock - 2 : 0);
            float d = scale / typical * (0.5f + (rng.Next() >> 8) / 16777216f);
            ushort bits = BitConverter.HalfToUInt16Bits((System.Half)d);
            raw[at] = (byte)(bits & 0xFF);
            raw[at + 1] = (byte)(bits >> 8);
            if (type is GgmlType.Q5_1 or GgmlType.Q4_K)
            {
                // Both formats have a second floating-point coefficient. Random
                // payload bytes are valid codes, but those coefficients must be finite.
                float minimum = type == GgmlType.Q5_1 ? -15.5f * d : 7.5f * d;
                ushort minBits = BitConverter.HalfToUInt16Bits((System.Half)minimum);
                raw[at + 2] = (byte)minBits;
                raw[at + 3] = (byte)(minBits >> 8);
            }
        }
        var ul = new ulong[dims.Length];
        for (int i = 0; i < dims.Length; i++) ul[i] = (ulong)dims[i];
        return new Tensor { Name = name, Dims = ul, Type = type, RawBlocks = raw };
    }

    /// <summary>
    /// ggml's <c>quantize_row_q8_0</c>, reproduced so the file is byte-identical to what
    /// <c>llama-quantize ... Q8_0</c> writes: per 32 values, scale = amax/127 stored as F16, then
    /// round-to-nearest of x/scale.
    /// </summary>
    public static byte[] QuantizeQ8_0(float[] data, int rowLength)
    {
        int rows = data.Length / rowLength;
        int blocksPerRow = rowLength / Q8Block;
        var outBytes = new byte[(long)rows * blocksPerRow * Q8BlockBytes];
        int o = 0;
        for (int r = 0; r < rows; r++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int off = r * rowLength + b * Q8Block;
                float amax = 0;
                for (int j = 0; j < Q8Block; j++) amax = Math.Max(amax, Math.Abs(data[off + j]));
                float d = amax / 127.0f;
                ushort dHalf = BitConverter.HalfToUInt16Bits((System.Half)d);
                float dRound = (float)BitConverter.UInt16BitsToHalf(dHalf);
                float id = dRound != 0 ? 1.0f / dRound : 0.0f;
                outBytes[o++] = (byte)(dHalf & 0xFF);
                outBytes[o++] = (byte)(dHalf >> 8);
                for (int j = 0; j < Q8Block; j++)
                    outBytes[o++] = unchecked((byte)(sbyte)Math.Round(data[off + j] * id, MidpointRounding.AwayFromZero));
            }
        }
        return outBytes;
    }

    /// <summary>xorshift32: identical output in any language, which is the point.</summary>
    private struct Rng
    {
        private uint _s;
        public Rng(uint seed) { _s = seed == 0 ? 0x9E3779B9u : seed; }
        public uint Next()
        {
            _s ^= _s << 13;
            _s ^= _s >> 17;
            _s ^= _s << 5;
            return _s;
        }
        /// <summary>Sum of four uniforms, centred: a rough bell shape in [-2, 2].</summary>
        public float Normalish(float scale)
        {
            double acc = 0;
            for (int i = 0; i < 4; i++) acc += Next() / 4294967296.0;
            return (float)((acc - 2.0) * scale);
        }
    }

    private static uint Fnv1a(string text)
    {
        uint h = 0x811c9dc5;
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            h ^= b;
            h = unchecked(h * 0x01000193);
        }
        return h;
    }

    // --- metadata -------------------------------------------------------------

    public abstract class Kv
    {
        public string Key;
        public abstract void Write(BinaryWriter w);
    }

    public sealed class U32 : Kv { public uint V; public override void Write(BinaryWriter w) { w.Write(4u); w.Write(V); } }
    public sealed class F32 : Kv { public float V; public override void Write(BinaryWriter w) { w.Write(6u); w.Write(V); } }
    public sealed class Bool : Kv { public bool V; public override void Write(BinaryWriter w) { w.Write(7u); w.Write(V); } }
    public sealed class Str : Kv { public string V; public override void Write(BinaryWriter w) { w.Write(8u); WriteStr(w, V); } }

    public sealed class StrArr : Kv
    {
        public IReadOnlyList<string> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(8u); w.Write((ulong)V.Count);
            foreach (string s in V) WriteStr(w, s);
        }
    }

    public sealed class I32Arr : Kv
    {
        public IReadOnlyList<int> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(5u); w.Write((ulong)V.Count);
            foreach (int v in V) w.Write(v);
        }
    }

    public sealed class BoolArr : Kv
    {
        public IReadOnlyList<bool> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(7u); w.Write((ulong)V.Count);
            foreach (bool v in V) w.Write(v);
        }
    }

    public sealed class U32Arr : Kv
    {
        public IReadOnlyList<uint> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(4u); w.Write((ulong)V.Count);
            foreach (uint v in V) w.Write(v);
        }
    }

    public sealed class U64Arr : Kv
    {
        public IReadOnlyList<ulong> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(10u); w.Write((ulong)V.Count);
            foreach (ulong v in V) w.Write(v);
        }
    }

    public sealed class F32Arr : Kv
    {
        public IReadOnlyList<float> V;
        public override void Write(BinaryWriter w)
        {
            w.Write(9u); w.Write(6u); w.Write((ulong)V.Count);
            foreach (float v in V) w.Write(v);
        }
    }

    private static void WriteStr(BinaryWriter w, string s)
    {
        byte[] b = Encoding.UTF8.GetBytes(s);
        w.Write((ulong)b.Length);
        w.Write(b);
    }

    /// <summary>GPT-2's byte-level tokenizer over the first <paramref name="vocab"/> bytes (no
    /// merges): every token is one byte.</summary>
    public static void AddByteTokenizer(List<Kv> kv, int vocab, uint bos, uint eos)
    {
        var tokens = new List<string>(vocab);
        foreach (int b in ByteToUnicode(vocab)) tokens.Add(char.ConvertFromUtf32(b));
        var types = new int[vocab];
        Array.Fill(types, 1);
        kv.Add(new Str { Key = "tokenizer.ggml.model", V = "gpt2" });
        kv.Add(new Str { Key = "tokenizer.ggml.pre", V = "gpt-2" });
        kv.Add(new StrArr { Key = "tokenizer.ggml.tokens", V = tokens });
        kv.Add(new I32Arr { Key = "tokenizer.ggml.token_type", V = types });
        kv.Add(new StrArr { Key = "tokenizer.ggml.merges", V = Array.Empty<string>() });
        kv.Add(new U32 { Key = "tokenizer.ggml.bos_token_id", V = bos });
        kv.Add(new U32 { Key = "tokenizer.ggml.eos_token_id", V = eos });
        kv.Add(new Bool { Key = "tokenizer.ggml.add_bos_token", V = false });
        kv.Add(new Bool { Key = "tokenizer.ggml.add_eos_token", V = false });
    }

    /// <summary>GPT-2's byte-to-unicode mapping, restricted to the first <paramref name="count"/> bytes.</summary>
    private static IEnumerable<int> ByteToUnicode(int count)
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
        for (int i = 0; i < count; i++) yield return map[i];
    }

    // --- container ------------------------------------------------------------

    public static void Write(string path, List<Kv> kv, List<Tensor> tensors)
    {
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write);
        using (var head = new MemoryStream())
        {
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
                    w.Write((uint)t.Type);
                    w.Write(offset);
                    offset += (t.ByteCount + Alignment - 1) / Alignment * Alignment;
                }
            }
            byte[] headBytes = head.ToArray();
            fs.Write(headBytes, 0, headBytes.Length);
            int pad = (Alignment - headBytes.Length % Alignment) % Alignment;
            for (int i = 0; i < pad; i++) fs.WriteByte(0);
        }

        var zeros = new byte[Alignment];
        foreach (var t in tensors)
        {
            byte[] raw = t.Raw();
            fs.Write(raw, 0, raw.Length);
            int tailPad = (Alignment - raw.Length % Alignment) % Alignment;
            if (tailPad > 0) fs.Write(zeros, 0, tailPad);
        }
    }
}
