// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Validation;
using R = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private static IEnumerable<Case> SelectorAndRingCases()
    {
        foreach (int vocabulary in new[] { 16, 127, 2049, 248320 }) foreach (int columns in new[] { 1, 7, 56 })
        {
            ushort[] logits = Values(vocabulary * columns, 27);
            // Equal high logits cross warp, CTA, m128 and vocabulary tail boundaries.
            for (int t = 0; t < columns; t++)
            {
                foreach (int id in new[] { 0, 1, 15, 16, 31, 32, 63, 64, 127, 128, 2047, 2048, vocabulary - 1 }.Where(x => x < vocabulary)) logits[t * vocabulary + id] = R.BFloat(4);
                logits[t * vocabulary + (vocabulary - 1 - t % Math.Min(vocabulary, 11))] = R.BFloat(8);
            }
            var expected = R.Top16(logits, vocabulary, columns);
            long scratchBytes = TensorSharp.Cuda.CudaNinferDraftModule.Top16WorkspaceBytes(vocabulary, columns);
            yield return new($"top16-v{vocabulary}-c{columns}", new() { ["logits"] = Bytes(logits) },
                new() { ["ids"] = new(Bytes(expected.Ids)), ["unary"] = new(Bytes(expected.Values)), ["scratch"] = new(null, ScratchBytes: checked((int)scratchBytes)) },
                (m, p) => m.Top16(p["logits"], p["ids"], p["unary"], vocabulary, columns, p["scratch"], IntPtr.Zero),
                new { oracle = "Full finite BF16 vocabulary ordering: descending value, ties smaller original row; unary exact F32 widening", exactBits = true, scratchContract = "guards only; internal partial selection layout is not an output oracle" });
        }
        foreach (int positions in new[] { 1, 3, 7 }) foreach (int lanes in new[] { 1, 2, 8 })
        {
            const int vocabulary = 512, rank = 256;
            int[] candidates = Enumerable.Range(0, 16 * positions * lanes).Select(i => (i * 37 + i / 16 * 13 + 127) % vocabulary).ToArray();
            int[] anchors = Enumerable.Range(0, lanes).Select(i => (i * 129 + 31) % vocabulary).ToArray();
            int[] predecessors = R.Predecessors(candidates, anchors, positions, lanes);
            float[] unary = Enumerable.Range(0, candidates.Length).Select(i => (i % 16 + 1) / 16f).ToArray();
            float[] hidden = Enumerable.Range(0, positions * lanes * rank).Select(i => ((i * 7 + i / rank * 11) % 13 - 6) / 8f).ToArray();
            byte[] sc = new byte[vocabulary * rank / 2], pc = new byte[sc.Length], ss = new byte[vocabulary * rank / 16], ps = new byte[ss.Length];
            for (int row = 0; row < vocabulary; row++) for (int c = 0; c < rank; c += 2)
            {
                sc[row * rank / 2 + c / 2] = (byte)(EncodedCode(row, c) | EncodedCode(row, c + 1) << 4);
                pc[row * rank / 2 + c / 2] = (byte)(EncodedCode(row + 3, c + 5) | EncodedCode(row + 3, c + 6) << 4);
            }
            int at = 0;
            for (int tile = 0; tile < vocabulary / 128; tile++) for (int kg = 0; kg < rank / 64; kg++) for (int r = 0; r < 32; r++) for (int quarter = 0; quarter < 4; quarter++) for (int g = 0; g < 4; g++)
            { int row = tile * 128 + quarter * 32 + r, group = kg * 4 + g; ss[at] = (byte)(48 + (row + group) % 3 * 8); ps[at++] = (byte)(48 + (row * 2 + group + 1) % 3 * 8); }
            float[] scores = new float[256 * positions * lanes];
            for (int slot = 0; slot < positions * lanes; slot++) for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++)
            {
                double sum = 0;
                for (int r = 0; r < rank; r++)
                {
                    int sr = candidates[slot * 16 + i], pr = predecessors[slot * 16 + j];
                    double a = R.Code((sc[sr * rank / 2 + r / 2] >> (r % 2 * 4)) & 15) * R.Scale(ss[R.ScaleOffset(sr, r / 16, rank)]);
                    double b = R.Code((pc[pr * rank / 2 + r / 2] >> (r % 2 * 4)) & 15) * R.Scale(ps[R.ScaleOffset(pr, r / 16, rank)]) * 2;
                    sum += a * b * hidden[slot * rank + r];
                }
                // All operands are small binary fractions/powers of two; every
                // intermediate product and reduction is exactly F32 representable.
                float exact = (float)sum + unary[slot * 16 + i];
                Need((double)(float)sum == sum, "Selector analytical fixture is not exactly representable.");
                scores[slot * 256 + j * 16 + i] = exact;
            }
            int[] walked = R.Walk(scores, candidates, positions, lanes);
            yield return new($"selector-compose-p{positions}-l{lanes}", new() { ["candidates"] = Bytes(candidates), ["anchors"] = Bytes(anchors), ["unary"] = Bytes(unary), ["hidden"] = Bytes(hidden),
                    ["sc"] = sc, ["ss"] = ss, ["pc"] = pc, ["ps"] = ps },
                new() { ["pred"] = new(Bytes(predecessors)), ["scores"] = new(Bytes(scores)), ["walked"] = new(Bytes(walked)) },
                (m, p) => { m.SelectorPredecessors(p["candidates"], p["anchors"], p["pred"], positions, lanes, IntPtr.Zero);
                    m.SelectorScoresRaw(p["candidates"], p["pred"], p["unary"], p["hidden"], p["sc"], p["ss"], 1, p["pc"], p["ps"], 2, p["scores"], vocabulary, positions, lanes, IntPtr.Zero);
                    m.SelectorWalk(p["scores"], p["candidates"], p["walked"], positions, lanes, IntPtr.Zero); },
                new { oracle = "Independent original code/scale decode, exactly represented rank256 binary products, predecessor lane-major and greedy strict-greater walk", exactBits = true, composedGPUCalls = 3 });
        }
        foreach (int width in new[] { 1, 7, 8, 64, 2048 }) foreach (int first in new[] { 0, 2047, 2048, 4095 })
        {
            const int batch = 2, capacity = 3;
            var key = Values(batch * width * 1024, 31); var value = Values(key.Length, 37);
            var ringK = Values(capacity * 2048 * 1024, 41); var ringV = Values(ringK.Length, 43);
            var expectedK = (ushort[])ringK.Clone(); var expectedV = (ushort[])ringV.Clone();
            int[] physical = { 2, 0 }, counts = { width, width == 1 ? 0 : width - 1 };
            int[] positions = Enumerable.Range(0, batch * width).Select(i => first + i % width + i / width * 3).ToArray();
            for (int l = 0; l < batch; l++) for (int t = 0; t < counts[l]; t++) for (int head = 0; head < 8; head++) for (int d = 0; d < 128; d++)
            {
                int source = (l * width + t) * 1024 + head * 128 + d;
                int destination = ((physical[l] * 8 + head) * 2048 + (positions[l * width + t] & 2047)) * 128 + d;
                expectedK[destination] = key[source]; expectedV[destination] = value[source];
            }
            yield return new($"ring-append-w{width}-p{first}", new() { ["key"] = Bytes(key), ["value"] = Bytes(value), ["positions"] = Bytes(positions), ["counts"] = Bytes(counts), ["lanes"] = Bytes(physical), ["ringK"] = Bytes(ringK), ["ringV"] = Bytes(ringV) },
                new() { ["ringK"] = new(Bytes(expectedK)), ["ringV"] = new(Bytes(expectedV)) },
                (m, p) => m.RingAppend(p["key"], p["value"], p["positions"], p["counts"], p["lanes"], p["ringK"], p["ringV"], width, batch, capacity, IntPtr.Zero),
                new { oracle = "CPU exact head-major physical-slot cyclic2048 scatter; every unwritten ring byte retained", exactBits = true, counts, physicalSlots = physical });
        }
    }
}
