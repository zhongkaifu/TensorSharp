// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;

internal static partial class Program
{
    private const long PlaneBytes = 4L * 256 * 256 * 2, KOffset = 64, VOffset = KOffset + PlaneBytes + 128;
    private const int BlockBodyBytes = (int)(VOffset + PlaneBytes + 64);
    private static int[] Permutation(int count, uint seed)
    {
        var values = Enumerable.Range(0, count).ToArray();
        for (int i = count - 1; i > 0; i--) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; int j = (int)(seed % (uint)(i + 1)); (values[i], values[j]) = (values[j], values[i]); }
        return values;
    }
    private sealed record Cache(Memory.Buffer Table, int Blocks, int[] BlockPermutation);
    private static Cache MakeCache(Memory memory, R.Fixture fixture)
    {
        // Same frozen R20 TS256 fixture construction, without unused original64 planes.
        int keys = fixture.Case.Keys, count = (keys + 255) / 256 + 1;
        int[] permutation = Permutation(count, 0x65e19bd3U), inverse = new int[count];
        for (int logical = 0; logical < count; logical++) inverse[permutation[logical]] = logical;
        var blocks = new Memory.Buffer[count];
        for (int physical = 0; physical < count; physical++)
        {
            int logical = inverse[physical]; var data = new byte[BlockBodyBytes]; Array.Fill(data, (byte)0xa5);
            foreach (long offset in new[] { KOffset, VOffset })
            {
                var words = MemoryMarshal.Cast<byte, ushort>(data.AsSpan((int)offset, (int)PlaneBytes)); words.Fill(0x7fc1);
                ushort[] source = offset == KOffset ? fixture.LogicalK : fixture.LogicalV;
                for (int t = 0; t < 256 && logical * 256 + t < keys; t++) for (int h = 0; h < 4; h++)
                    source.AsSpan(((logical * 256 + t) * 4 + h) * 256, 256).CopyTo(words.Slice((h * 256 + t) * 256, 256));
            }
            blocks[physical] = memory.Put("ts256-physical-" + physical, data);
        }
        var table = memory.Put("ts256-pointer-table", Bytes(Enumerable.Range(0, count).Select(i => blocks[permutation[i]].Pointer.ToInt64()).ToArray()));
        return new(table, count, permutation);
    }

    // Identical finite-value/analytic checks and fixed bounds to the frozenR20
    // CheckGated. Exact GPU control bits are a separate, mandatory stronger gate.
    private sealed record Numeric(double MaxAbsolute, double MaxScaledError, int Nonfinite, int Violations, int ExactAnalyticViolations);
    private static Numeric CheckGated(byte[] body, R.Fixture fixture)
    {
        var data = MemoryMarshal.Cast<byte, float>(body); double max = 0, scaled = 0; int nonfinite = 0, violations = 0, exact = 0;
        Need(data.Length == fixture.IdealGated.Length, "Gated output length differs.");
        for (int i = 0; i < data.Length; i++)
        {
            double actual = data[i], expected = fixture.IdealGated[i], error = Math.Abs(actual - expected), bound = R.Absolute + R.Relative * Math.Abs(expected);
            if (!double.IsFinite(actual)) { nonfinite++; continue; } if (!(error <= bound)) violations++; max = Math.Max(max, error); scaled = Math.Max(scaled, error / bound);
            int d = i % 256, h = i / 256 % 24, t = i / 6144;
            if (fixture.Case.Pattern == "constant" || fixture.Case.Pattern == "sentinel" && d % 17 != 0)
            { if (BitConverter.SingleToInt32Bits(data[i]) != BitConverter.SingleToInt32Bits(R.Constant(h / 6, d) * .5f)) exact++; }
            else if (fixture.Case.Pattern == "sentinel" && fixture.Case.Past + t < fixture.Case.Sentinel && data[i] != 0) exact++;
        }
        return new(max, scaled, nonfinite, violations, exact);
    }
    private static bool Pass(Numeric n) => n.Nonfinite == 0 && n.Violations == 0 && n.ExactAnalyticViolations == 0;

    private static void RunCase(LegacyAttention module, R.Fixture f, int lifetime, string output,
        List<Dictionary<string, object?>> records, Dictionary<string, string> first,
        Dictionary<string, string> files, Action checkpoint)
    {
        using var memory = new Memory(); var c = f.Case; int elements = c.Rows * 6144;
        var q = memory.Put("Q-F32", Bytes(f.QueryF32)); var gate = memory.Put("gate-F32", Bytes(f.Gate));
        var ends = memory.Put("key-ends", Bytes(Enumerable.Range(0, c.Rows).Select(i => c.Past + i + 1).ToArray()));
        var cache = MakeCache(memory, f);
        var control = memory.Empty("control256", elements * 4L); var candidate = memory.Empty("candidate128", elements * 4L);
        var passes = new List<object>(); var record = new Dictionary<string, object?> { ["id"] = $"{c.Name}/life{lifetime}", ["case"] = c.Name, ["lifetime"] = lifetime,
            ["rows"] = c.Rows, ["past"] = c.Past, ["controlThreads"] = 256, ["candidateThreads"] = 128,
            ["controlGridX"] = (c.Rows * 6 + 127) / 128, ["candidateGridX"] = (c.Rows * 6 + 63) / 64,
            ["gridY"] = 4, ["gridZ"] = 1, ["dynamicSharedBytes"] = 0,
            ["blockPermutation"] = cache.BlockPermutation, ["kOffset"] = KOffset, ["vOffset"] = VOffset, ["blockBodyBytes"] = BlockBodyBytes,
            ["inputAllocationSha256"] = memory.Buffers.Where(x => x.ImmutableSha != null).ToDictionary(x => x.Name, x => x.ImmutableSha),
            ["passed"] = false, ["passes"] = passes }; records.Add(record); checkpoint();
        for (int pass = 0; pass < 3; pass++)
        {
            Memory.Poison(control, Poisons[pass]); Memory.Poison(candidate, Poisons[pass]);
            module.Run(q.Pointer, gate.Pointer, ends.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, 256, control.Pointer);
            module.Run(q.Pointer, gate.Pointer, ends.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, 128, candidate.Pointer);
            byte[] controlBytes = Memory.Body(control), candidateBytes = Memory.Body(candidate);
            bool exact = candidateBytes.AsSpan().SequenceEqual(controlBytes);
            Numeric controlNumeric = CheckGated(controlBytes, f), candidateNumeric = CheckGated(candidateBytes, f);
            string controlFile = Cas(controlBytes, output, files), candidateFile = Cas(candidateBytes, output, files);
            bool aa = true;
            foreach (var item in new[] { ("control256", controlBytes, controlFile), ("candidate128", candidateBytes, candidateFile) })
            {
                string key = c.Name + "/" + item.Item1;
                if (first.TryGetValue(key, out var previous)) aa &= File.ReadAllBytes(Path.Combine(output, previous)).AsSpan().SequenceEqual(item.Item2);
                else first.Add(key, item.Item3);
            }
            bool immutable = memory.CheckInputs(), guards = memory.CheckGuards();
            bool ok = exact && Pass(controlNumeric) && Pass(candidateNumeric) && aa && immutable && guards;
            passes.Add(new { pass, poisonByte = Poisons[pass], controlFile, candidateFile, controlBodySha256 = Sha(controlBytes), candidateBodySha256 = Sha(candidateBytes),
                fullF32BitsEqual = exact, controlFp64 = controlNumeric, candidateFp64 = candidateNumeric, aaAndCrossLifetimeBits = aa, immutableInputs = immutable, allAllocationGuards = guards, passed = ok });
            checkpoint(); Need(ok, "Same-image scheduling gate failed; complete candidate/control raw bodies retained inCAS.");
        }
        Need(module.KernelSubmissions == 6, "Actual host submission coverage differs.");
        record["kernelSubmissions"] = module.KernelSubmissions; record["passed"] = true; checkpoint();
    }
}
