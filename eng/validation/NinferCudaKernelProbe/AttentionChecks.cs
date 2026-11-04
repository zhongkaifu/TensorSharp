// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

internal static class AttentionChecks
{
    private const int Hd = 256, Heads = 24, KvHeads = 4, Block = 256;
    // Declared before device evaluation: normalization/RoPE allows at most small
    // BF16 rounding differences; attention compares against FP64 ideal softmax,
    // rather than copying the CUDA tile algorithm or tuning to its output.
    private const float PrepareAbsolute = 0.015625f;
    private const float AttentionAbsolute = 0.015625f, AttentionRelative = 0.005f;
    private static ushort Bits(float x)
    {
        uint u = unchecked((uint)BitConverter.SingleToInt32Bits(x));
        return (ushort)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
    }
    private static float Value(ushort x) => BitConverter.Int32BitsToSingle(x << 16);
    private static float Bf(float x) => Value(Bits(x));
    private static void Need(bool value, string message) { if (!value) throw new InvalidOperationException(message); }
    private static object Check(string name, float[] actual, float[] expected, float absolute, float relative)
    {
        Need(actual.Length == expected.Length, name + ": size mismatch");
        int mismatches = 0; double max = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            Need(float.IsFinite(actual[i]) && float.IsFinite(expected[i]), name + ": nonfinite value");
            double delta = Math.Abs((double)actual[i] - expected[i]); max = Math.Max(max, delta);
            if (absolute == 0 && relative == 0
                ? BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i])
                : delta > absolute + relative * Math.Abs(expected[i])) mismatches++;
        }
        Need(mismatches == 0, $"{name}: {mismatches} mismatches, max {max}");
        return new { name, elements = actual.Length, absolute, relative, maxAbsoluteError = max, mismatches, passed = true };
    }
    private sealed class Cache : IDisposable
    {
        public readonly Memory Memory = new();
        public readonly int Blocks;
        public readonly IntPtr Table;
        public readonly IntPtr[] Pointers;
        public readonly ushort[][] Host;
        public const int PlaneElements = KvHeads * Block * Hd;
        public const long PlaneBytes = PlaneElements * 2L;
        public Cache(int sequences, int keys, bool populated)
        {
            Blocks = (keys + Block - 1) / Block;
            Host = Enumerable.Range(0, sequences * Blocks).Select(_ => new ushort[2 * PlaneElements]).ToArray();
            if (populated)
                for (int seq = 0; seq < sequences; seq++)
                    for (int key = 0; key < keys; key++)
                        for (int h = 0; h < KvHeads; h++)
                            for (int d = 0; d < Hd; d++)
                            {
                                Set(seq, key, h, d, false, Key(seq, key, h, d));
                                Set(seq, key, h, d, true, V(seq, key, h, d));
                            }
            Pointers = Host.Select(Memory.Put).ToArray(); Table = Memory.Put(Pointers);
        }
        private int Offset(int key, int head, int d, bool v) => (v ? PlaneElements : 0) + (head * Block + key % Block) * Hd + d;
        public void Set(int seq, int key, int head, int d, bool v, float value) => Host[seq * Blocks + key / Block][Offset(key, head, d, v)] = Bits(value);
        public float Get(int seq, int key, int head, int d, bool v) => Value(Host[seq * Blocks + key / Block][Offset(key, head, d, v)]);
        public void Dispose() => Memory.Dispose();
    }
    private static float Key(int seq, int key, int head, int d) => Bf(((key * 7 + d * 11 + head * 13 + seq * 3) % 31 - 15) / 32f);
    private static float V(int seq, int key, int head, int d) => Bf(((key * 3 + d * 5 + head * 7 + seq * 11) % 31 - 15) / 16f);
    private static float Query(int seq, int row, int head, int d) => Bf(((d * 17 + head * 13 + row * 7 + seq * 3) % 65 - 32) / 32f);

    public static List<object> Run()
    {
        using var kernels = new CudaNinferAttention();
        var checks = new List<object>();
        foreach (int rows in new[] { 1, 7, 8, 9, 33 }) Prepare(kernels, rows, rows == 7 ? 255 : 0, checks);
        foreach (var (rows, past) in new[] { (1, 0), (7, 0), (8, 0), (9, 0), (33, 0), (7, 255), (1, 1025) })
        {
            Attention(kernels, rows, past, false, checks);
            if (rows * (Heads / KvHeads) <= Q35Kernels.AttnBlockRows) Attention(kernels, rows, past, true, checks);
        }
        return checks;
    }
    private static void Prepare(CudaNinferAttention kernels, int rows, int past, List<object> checks)
    {
        const int sequences = 2, packedWidth = 14336;
        using var cache = new Cache(sequences, past + rows, true); using var memory = new Memory();
        int total = rows * sequences;
        float[] packed = new float[total * packedWidth], norm = Enumerable.Repeat(1f, Hd).ToArray();
        float[] wantQ = new float[total * Heads * Hd], wantGate = new float[wantQ.Length];
        int[] positions = new int[total * 4], cacheRows = new int[total];
        for (int s = 0; s < sequences; s++) for (int t = 0; t < rows; t++)
        {
            int row = s * rows + t;
            cacheRows[row] = past + t;
            for (int axis = 0; axis < 4; axis++) positions[axis * total + row] = past + t;
            for (int h = 0; h < Heads + 2 * KvHeads; h++)
            {
                bool isQ = h < Heads, isK = h >= Heads && h < Heads + KvHeads;
                int head = isQ ? h : isK ? h - Heads : h - Heads - KvHeads;
                int offset = row * packedWidth + (isQ ? head * 2 * Hd : Heads * 2 * Hd + (h - Heads) * Hd);
                float[] values = new float[Hd];
                for (int d = 0; d < Hd; d++)
                {
                    // Constant squared magnitude gives an exactly representable RMS;
                    // varying signs and head gates still expose row/head permutations.
                    packed[offset + d] = values[d] = ((d + head + t + s) & 1) == 0 ? 1f : -1f;
                    if (isQ) packed[offset + Hd + d] = (head % 5 - 2) * 0.5f;
                }
                if (isQ || isK)
                {
                    float inv = 1f / MathF.Sqrt(1f + 1e-6f);
                    for (int d = 0; d < Hd; d++) values[d] = Bf(values[d] * inv);
                    for (int pair = 0; pair < 32; pair++)
                    {
                        // This component contract matches the declared ordinary text
                        // theta=1e7 frequency. Original-engine precomputed tables need
                        // an additional independent full-model oracle comparison.
                        float angle = (past + t) * MathF.Pow(10000000f, -pair / 32f);
                        float sn = MathF.Sin(angle), cs = MathF.Cos(angle);
                        float x = values[pair], y = values[pair + 32];
                        values[pair] = Bf(x * cs - y * sn); values[pair + 32] = Bf(x * sn + y * cs);
                    }
                }
                for (int d = 0; d < Hd; d++)
                {
                    if (isQ) { int at = (row * Heads + head) * Hd + d; wantQ[at] = values[d]; wantGate[at] = packed[offset + Hd + d]; }
                    else cache.Set(s, past + t, head, d, !isK, values[d]);
                }
            }
        }
        var q = memory.Zeros<float>(wantQ.Length); var gate = memory.Zeros<float>(wantGate.Length);
        kernels.Prepare(memory.Put(packed), total, rows, Heads, KvHeads, memory.Put(norm), memory.Put(norm), 1e-6f,
            memory.Put(positions), total, 11, 11, 10, 10000000f, memory.Put(cacheRows), cache.Table, cache.Blocks,
            0, Cache.PlaneBytes, q, gate, IntPtr.Zero);
        string name = $"attention_prepare_t{rows}_past{past}";
        checks.Add(Check(name + "_q", memory.Read<float>(q, wantQ.Length), wantQ, PrepareAbsolute, 0));
        checks.Add(Check(name + "_gate", memory.Read<float>(gate, wantGate.Length), wantGate, 0, 0));
        int changed = 0; double max = 0;
        for (int b = 0; b < cache.Pointers.Length; b++)
        {
            ushort[] actual = cache.Memory.Read<ushort>(cache.Pointers[b], cache.Host[b].Length);
            for (int i = 0; i < actual.Length; i++)
            {
                int key = (b % cache.Blocks) * Block + (i % Cache.PlaneElements / Hd) % Block;
                bool normWrite = i < Cache.PlaneElements && key >= past && key < past + rows;
                double delta = Math.Abs((double)Value(actual[i]) - Value(cache.Host[b][i])); max = Math.Max(max, delta);
                if (normWrite ? delta > PrepareAbsolute : actual[i] != cache.Host[b][i]) changed++;
            }
        }
        Need(changed == 0, name + ": incorrect KV values or writes outside declared cache rows");
        checks.Add(new { name = name + "_kv_and_untouched_rows", blocks = cache.Pointers.Length, absoluteOnNewK = PrepareAbsolute, allOtherBitsExact = true, maxAbsoluteError = max, passed = true });
    }
    private static void Attention(CudaNinferAttention kernels, int rows, int past, bool split, List<object> checks)
    {
        const int sequences = 2;
        using var cache = new Cache(sequences, past + rows, true); using var memory = new Memory();
        int count = sequences * rows * Heads * Hd;
        float[] query = new float[count], gate = new float[count], expected = new float[count];
        int[] ends = new int[sequences * rows];
        for (int s = 0; s < sequences; s++) for (int t = 0; t < rows; t++)
        {
            int row = s * rows + t, keys = past + t + 1; ends[row] = keys;
            for (int h = 0; h < Heads; h++)
            {
                int head = h / (Heads / KvHeads), at = (row * Heads + h) * Hd;
                for (int d = 0; d < Hd; d++) { query[at + d] = Query(s, t, h, d); gate[at + d] = (h % 5 - 2) * 0.5f; }
                double[] scores = new double[keys];
                for (int key = 0; key < keys; key++)
                {
                    double dot = 0; for (int d = 0; d < Hd; d++) dot += (double)query[at + d] * cache.Get(s, key, head, d, false);
                    scores[key] = dot / Math.Sqrt(Hd);
                }
                double max = scores.Max(), den = 0;
                for (int key = 0; key < keys; key++) den += scores[key] = Math.Exp(scores[key] - max);
                for (int d = 0; d < Hd; d++)
                {
                    double sum = 0; for (int key = 0; key < keys; key++) sum += scores[key] * cache.Get(s, key, head, d, true);
                    expected[at + d] = Bf(Bf((float)(sum / den)) * (1f / (1f + MathF.Exp(-gate[at + d]))));
                }
            }
        }
        var q = memory.Put(query); var gates = memory.Put(gate); var keyEnds = memory.Put(ends);
        float[] poison = Enumerable.Repeat(float.NaN, count + 512).ToArray();
        for (int i = 0; i < 256; i++) poison[i] = poison[count + 256 + i] = 123.125f;
        var guarded = memory.Put(poison); var output = guarded + 256 * 4;
        var plan = Q35Kernels.GetSplitAttentionPlan(past + rows);
        var partial = split ? memory.Zeros<float>(checked(sequences * rows * Heads * plan.Splits * (Hd + 2))) : IntPtr.Zero;
        kernels.Attention(q, gates, keyEnds, cache.Table, cache.Blocks, 0, Cache.PlaneBytes,
            rows, sequences, Heads, KvHeads, 1f / MathF.Sqrt(Hd), past + rows, partial, output, IntPtr.Zero);
        float[] actual = memory.Read<float>(output, count);
        string name = $"attention_fp64_softmax_t{rows}_past{past}_{(split ? "split" : "direct")}";
        checks.Add(Check(name, actual, expected, AttentionAbsolute, AttentionRelative));
        float[] all = memory.Read<float>(guarded, count + 512);
        Need(all.Take(256).All(x => x == 123.125f) && all.Skip(count + 256).All(x => x == 123.125f), name + ": output guard overwritten");
        kernels.Attention(q, gates, keyEnds, cache.Table, cache.Blocks, 0, Cache.PlaneBytes,
            rows, sequences, Heads, KvHeads, 1f / MathF.Sqrt(Hd), past + rows, partial, output, IntPtr.Zero);
        checks.Add(Check(name + "_repeat_fullbits", memory.Read<float>(output, count), actual, 0, 0));
    }
}
