// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.Cuda;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;

internal static partial class Program
{
    private const long KOffset = 64, VOffset = KOffset + CudaNinferPrefillAttentionModule.KvPlaneBlockBytes + 128;
    private const int BlockBodyBytes = (int)(VOffset + CudaNinferPrefillAttentionModule.KvPlaneBlockBytes + 64);
    private static int[] Permutation(int count, uint seed)
    {
        var values = Enumerable.Range(0, count).ToArray();
        for (int i = count - 1; i > 0; i--) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; int j = (int)(seed % (uint)(i + 1)); (values[i], values[j]) = (values[j], values[i]); }
        return values;
    }
    private sealed record Cache(Memory.Buffer Table, Memory.Buffer OriginalK, Memory.Buffer OriginalV,
        Memory.Buffer OriginalTable, Memory.Buffer Positions, int Blocks, int[] BlockPermutation, int[] OriginalPermutation);
    private static Cache MakeCache(Memory memory, R.Fixture fixture)
    {
        int keys = fixture.Case.Keys, count = (keys + 255) / 256 + 1, pages = (keys + 63) / 64 + 1;
        int[] permutation = Permutation(count, 0x65e19bd3U), originalPermutation = Permutation(pages, 0x71c42fe9U);
        int[] inversePermutation = new int[count];
        for (int logical = 0; logical < count; logical++) inversePermutation[permutation[logical]] = logical;
        var physicalBlocks = new Memory.Buffer[count];
        // Allocate in physical order. Allocating in logical order and merely
        // permuting labels would cancel the intended pointer-table challenge.
        for (int physical = 0; physical < count; physical++)
        {
            int logical = inversePermutation[physical];
            var data = new byte[BlockBodyBytes]; Array.Fill(data, (byte)0xa5);
            foreach (long offset in new[] { KOffset, VOffset })
            {
                var words = MemoryMarshal.Cast<byte, ushort>(data.AsSpan((int)offset, (int)CudaNinferPrefillAttentionModule.KvPlaneBlockBytes)); words.Fill(0x7fc1);
                ushort[] source = offset == KOffset ? fixture.LogicalK : fixture.LogicalV;
                for (int t = 0; t < 256 && logical * 256 + t < keys; t++) for (int h = 0; h < 4; h++)
                    source.AsSpan(((logical * 256 + t) * 4 + h) * 256, 256).CopyTo(words.Slice((h * 256 + t) * 256, 256));
            }
            physicalBlocks[physical] = memory.Put("ts256-physical-" + physical, data);
        }
        var table = memory.Put("ts256-pointer-table", Bytes(Enumerable.Range(0, count).Select(i => physicalBlocks[permutation[i]].Pointer.ToInt64()).ToArray()));
        ushort[] BuildOriginal(ushort[] source)
        {
            var data = new ushort[pages * 4 * 64 * 256]; Array.Fill(data, (ushort)0x7fc1);
            for (int key = 0; key < keys; key++) for (int h = 0; h < 4; h++)
                source.AsSpan((key * 4 + h) * 256, 256).CopyTo(data.AsSpan(((originalPermutation[key / 64] * 4 + h) * 64 + key % 64) * 256, 256));
            return data;
        }
        var originalK = memory.Put("original64-K", Bytes(BuildOriginal(fixture.LogicalK)));
        var originalV = memory.Put("original64-V", Bytes(BuildOriginal(fixture.LogicalV)));
        var originalTable = memory.Put("original64-I32-table", Bytes(originalPermutation));
        var positions = memory.Put("original-base-position", Bytes(new[] { fixture.Case.Past }));
        return new(table, originalK, originalV, originalTable, positions, count, permutation, originalPermutation);
    }
    private sealed record Numeric(double MaxAbsolute, double MaxScaledError, int Nonfinite, int Violations, int ExactAnalyticViolations);
    private static Numeric CheckCore(byte[] body, R.Fixture fixture)
    {
        var data = MemoryMarshal.Cast<byte, ushort>(body); double max = 0, scaled = 0; int nonfinite = 0, violations = 0, exact = 0;
        Need(data.Length == fixture.IdealCore.Length, "Core output length differs.");
        for (int i = 0; i < data.Length; i++)
        {
            double actual = R.Float(data[i]), expected = fixture.IdealCore[i], error = Math.Abs(actual - expected), bound = R.Absolute + R.Relative * Math.Abs(expected);
            if (!double.IsFinite(actual)) { nonfinite++; continue; } if (!(error <= bound)) violations++; max = Math.Max(max, error); scaled = Math.Max(scaled, error / bound);
            int d = i % 256, h = i / 256 % 24, t = i / 6144;
            if (fixture.Case.Pattern == "constant" || fixture.Case.Pattern == "sentinel" && d % 17 != 0)
            { if (data[i] != R.Bf(R.Constant(h / 6, d))) exact++; }
            else if (fixture.Case.Pattern == "sentinel" && fixture.Case.Past + t < fixture.Case.Sentinel && data[i] != 0) exact++;
        }
        return new(max, scaled, nonfinite, violations, exact);
    }
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

    private static void RunCase(CudaNinferPrefillAttentionModule module, LegacyAttention legacy, R.Fixture fixture, int lifetime,
        string output, List<Dictionary<string, object?>> records, Dictionary<string, string> repeat, Dictionary<string, string> files, Action checkpoint)
    {
        using var memory = new Memory(); var c = fixture.Case; int elements = c.Rows * 6144;
        var q = memory.Put("Q-BF16", Bytes(fixture.Query)); var q32 = memory.Put("Q-F32", Bytes(fixture.QueryF32)); var gate = memory.Put("gate-F32", Bytes(fixture.Gate));
        var ends = memory.Put("old-key-ends", Bytes(Enumerable.Range(0, c.Rows).Select(i => c.Past + i + 1).ToArray()));
        var cache = MakeCache(memory, fixture); var plan = CudaNinferPrefillAttentionModule.Plan(c.Rows, c.Keys, cache.Blocks, 170);
        Need(plan.Splits == c.ExpectedSplits, "Original split policy differs from the pinned170SM fixture.");
        var workspace = memory.Empty("candidate-workspace", plan.Bytes);
        var paged = memory.Empty("paged-core", elements * 2L); var original = memory.Empty("original-core", elements * 2L);
        var gated = memory.Empty("candidate-gated", elements * 4L); var old = memory.Empty("legacy-gated", elements * 4L);
        var acc = memory.Empty("original-partial-acc", (long)elements * plan.Splits * 4);
        var m = memory.Empty("original-partial-m", (long)c.Rows * 24 * plan.Splits * 4);
        var l = memory.Empty("original-partial-l", (long)c.Rows * 24 * plan.Splits * 4);
        var passes = new List<object>(); var record = new Dictionary<string, object?> { ["id"] = $"{c.Name}/life{lifetime}", ["case"] = c.Name, ["lifetime"] = lifetime, ["rows"] = c.Rows, ["past"] = c.Past, ["splits"] = plan.Splits, ["passed"] = false, ["passes"] = passes,
            ["blockPermutation"] = cache.BlockPermutation, ["originalPagePermutation"] = cache.OriginalPermutation, ["kOffset"] = KOffset, ["vOffset"] = VOffset, ["blockBodyBytes"] = BlockBodyBytes,
            ["inputAllocationSha256"] = memory.Buffers.Where(x => x.ImmutableSha != null).ToDictionary(x => x.Name, x => x.ImmutableSha) };
        records.Add(record); checkpoint();
        for (int pass = 0; pass < 3; pass++)
        {
            foreach (var b in memory.Buffers.Where(x => x.ImmutableSha == null)) Memory.Poison(b, Poisons[pass]);
            module.RunPagedBf16(q.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, c.Past, .0625f, paged.Pointer, workspace.Pointer, plan.Bytes, 170, IntPtr.Zero);
            module.RunOriginalBf16(q.Pointer, cache.OriginalK.Pointer, cache.OriginalV.Pointer, cache.OriginalTable.Pointer, cache.Positions.Pointer,
                original.Pointer, c.Rows, .0625f, plan.Splits, acc.Pointer, m.Pointer, l.Pointer, IntPtr.Zero);
            byte[] pagedBytes = Memory.Body(paged), originalBytes = Memory.Body(original);
            bool exactOriginal = pagedBytes.AsSpan().SequenceEqual(originalBytes); var coreNumeric = CheckCore(pagedBytes, fixture);
            bool partialsExact = true;
            if (plan.Splits > 1)
            {
                foreach (var item in new[] { (acc, plan.PartialAcc), (m, plan.PartialM), (l, plan.PartialL) })
                {
                    byte[] candidate = Memory.Read(workspace.Pointer + (nint)item.Item2, item.Item1.BodyBytes), reference = Memory.Body(item.Item1);
                    if (!candidate.AsSpan().SequenceEqual(reference))
                    {
                        partialsExact = false;
                        string stem = $"{c.Name}/life{lifetime}-pass{pass}-{item.Item1.Name}";
                        Capture(stem + "-candidate.bin", candidate, output, files); Capture(stem + "-original.bin", reference, output, files);
                    }
                }
            }
            Memory.Poison(workspace, Poisons[pass]);
            module.RunF32(q32.Pointer, gate.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, c.Past, .0625f,
                gated.Pointer, workspace.Pointer, plan.Bytes, 170, IntPtr.Zero);
            byte[] bridgeCore = Memory.Read(workspace.Pointer + (nint)plan.Output, elements * 2), packedQuery = Memory.Read(workspace.Pointer + (nint)plan.Query, elements * 2);
            legacy.Run(q32.Pointer, gate.Pointer, ends.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, old.Pointer);
            byte[] gatedBytes = Memory.Body(gated), oldBytes = Memory.Body(old); var gatedNumeric = CheckGated(gatedBytes, fixture); var oldNumeric = CheckGated(oldBytes, fixture);
            bool packedExact = packedQuery.AsSpan().SequenceEqual(Bytes(fixture.Query)), bridgeExact = bridgeCore.AsSpan().SequenceEqual(originalBytes);
            bool immutable = memory.CheckInputs(), guards = memory.CheckGuards(); bool repeated = true;
            var captured = new Dictionary<string, string>();
            foreach (var item in new[] { ("paged-core", pagedBytes), ("original-core", originalBytes), ("f32-bridge-core", bridgeCore), ("candidate-gated", gatedBytes), ("legacy-gated", oldBytes) })
            {
                string key = c.Name + "/" + item.Item1, hash = Sha(item.Item2); if (repeat.TryGetValue(key, out var expected) && expected != hash) repeated = false; repeat.TryAdd(key, hash);
                string file = $"{c.Name}/life{lifetime}-pass{pass}-{item.Item1}.bin"; Capture(file, item.Item2, output, files); captured.Add(item.Item1, hash);
            }
            int oldDifferent = 0; var newWords = MemoryMarshal.Cast<byte, uint>(gatedBytes); var oldWords = MemoryMarshal.Cast<byte, uint>(oldBytes);
            for (int i = 0; i < newWords.Length; i++) if (newWords[i] != oldWords[i]) oldDifferent++;
            bool ok = exactOriginal && partialsExact && packedExact && bridgeExact && immutable && guards && repeated && Pass(coreNumeric) && Pass(gatedNumeric) && Pass(oldNumeric);
            passes.Add(new { pass, poisonByte = Poisons[pass], originalCoreBitExact = exactOriginal, originalPartialBitsExact = partialsExact, packedQueryBitsExact = packedExact, bridgeCoreBitsExact = bridgeExact,
                coreFp64 = coreNumeric, gatedFp64 = gatedNumeric, legacyGatedFp64 = oldNumeric, newVsLegacyDifferentF32 = oldDifferent, immutableInputs = immutable, allAllocationGuards = guards, aaAndCrossLifetime = repeated, captured, passed = ok }); checkpoint();
            Need(ok, "Prefill attention gate failed; full core/gated outputs and any differing partials retained.");
        }
        Need(module.PagedCoreLaunches == 6 && module.OriginalCoreLaunches == 3, "Actual core submission coverage differs.");
        record["pagedCoreSubmissions"] = module.PagedCoreLaunches; record["originalCoreSubmissions"] = module.OriginalCoreLaunches; record["passed"] = true; checkpoint();
    }
    private static void Capture(string file, byte[] data, string output, Dictionary<string, string> files)
    { Need(!files.ContainsKey(file), "Duplicate evidence file."); NewFile(Path.Combine(output, file), data); files.Add(file, Sha(data)); }
}
