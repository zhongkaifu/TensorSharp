// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using R = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private static IEnumerable<Case> SwaCases()
    {
        foreach (bool dense in new[] { false, true }) foreach (int block in dense ? new[] { 7 } : new[] { 1, 7, 8 })
            foreach (int history in dense ? new[] { 0, 97, 2049 } : new[] { 0, 1, 95, 96, 97, 2047, 2048, 2049, 250000 })
        {
            const int batch = 2, capacity = 3;
            int maxContext = history; var layout = CudaNinferDraftModule.ValidateSwa(maxContext, block, batch, capacity);
            int[] physical = { 2, 0 }, valid = { block, Math.Max(1, block - 1) };
            int[] positions = Enumerable.Range(0, batch * block).Select(i => history + i % block).ToArray();
            ushort[] query = new ushort[batch * block * 4096], key = new ushort[batch * block * 1024], value = new ushort[key.Length];
            ushort[] ringK = Enumerable.Repeat(R.BFloat(7), capacity * 2048 * 1024).ToArray(), ringV = (ushort[])ringK.Clone();
            for (int l = 0; l < batch; l++)
            {
                for (int t = 0; t < block; t++)
                {
                    for (int qh = 0; qh < 32; qh++) for (int d = 0; d < 128; d++)
                        query[((l * block + t) * 32 + qh) * 128 + d] = dense ? R.BFloat((((t * 13 + qh * 7 + d * 3 + l) % 31) - 15) / 128f) : (ushort)0;
                    for (int kh = 0; kh < 8; kh++) for (int d = 0; d < 128; d++)
                    {
                        int at = ((l * block + t) * 8 + kh) * 128 + d;
                        key[at] = dense ? R.BFloat((((t * 5 + kh * 11 + d * 7 + l) % 31) - 15) / 128f) : (ushort)0;
                        value[at] = t < valid[l] ? R.BFloat(dense ? (((t * 7 + kh * 3 + d + l) % 31) - 15) / 16f : (t * 2 + kh + d + l * 7) % 3 - 1) : R.BFloat(9);
                    }
                }
                for (int p = history - Math.Min(history, 2047); p < history; p++) for (int kh = 0; kh < 8; kh++) for (int d = 0; d < 128; d++)
                {
                    int at = ((physical[l] * 8 + kh) * 2048 + (p & 2047)) * 128 + d;
                    ringK[at] = dense ? R.BFloat((((p * 5 + kh * 11 + d * 7 + l) % 31) - 15) / 128f) : (ushort)0;
                    ringV[at] = R.BFloat(dense ? (((p * 7 + kh * 3 + d + l) % 31) - 15) / 16f : (p + kh * 3 + d + l * 5) % 3 - 1);
                }
            }
            ushort[] expected = new ushort[query.Length]; double[] ideal = new double[query.Length], bound = new double[query.Length];
            for (int l = 0; l < batch; l++) for (int t = 0; t < valid[l]; t++) for (int qh = 0; qh < 32; qh++)
            {
                int kh = qh / 4, first = Math.Max(history - Math.Min(history, 2047), positions[l * block + t] - 2047);
                int count = history - first + valid[l]; double[] logits = new double[count];
                int KeyIndex(int j) => j < history - first ? ((physical[l] * 8 + kh) * 2048 + ((first + j) & 2047)) * 128 : ((l * block + j - (history - first)) * 8 + kh) * 128;
                bool IsHistory(int j) => j < history - first;
                if (dense) for (int j = 0; j < count; j++)
                {
                    var keys = IsHistory(j) ? ringK : key; int ki = KeyIndex(j), qi = ((l * block + t) * 32 + qh) * 128;
                    double dot = 0; for (int d = 0; d < 128; d++) dot += (double)R.Float(query[qi + d]) * R.Float(keys[ki + d]);
                    logits[j] = dot * .08838834764831845f;
                }
                double maximum = logits.Max(), denominator = 0; double[] weights = new double[count];
                for (int j = 0; j < count; j++) { weights[j] = Math.Exp(logits[j] - maximum); denominator += weights[j]; }
                for (int d = 0; d < 128; d++)
                {
                    double numerator = 0;
                    for (int j = 0; j < count; j++) numerator += weights[j] * R.Float((IsHistory(j) ? ringV : value)[KeyIndex(j) + d]);
                    int at = ((l * block + t) * 32 + qh) * 128 + d;
                    ideal[at] = numerator / denominator;
                    if (dense)
                        // Scoped, declared-before-GPU envelope for |Q|,|K|<1/8,
                        // |V|<=1; covers original BF16 P/partial/output boundaries.
                        // It is not a general model-quality or universal error proof.
                        bound[at] = .015625 + .0078125 * Math.Abs(ideal[at]);
                    else
                    {
                        Need(numerator == (float)numerator && denominator == (float)denominator, "SWA integer analytic fixture is inexact.");
                        float average = layout.Direct ? (float)numerator * (1f / (float)denominator) : (float)numerator / (float)denominator;
                        expected[at] = R.BFloat(average);
                        bound[at] = R.BFloatUnitRoundoff * Math.Abs(ideal[at]) + 8 * R.FloatUnitRoundoff;
                    }
                }
            }
            var outputs = new Dictionary<string, Expected> { ["output"] = new(dense ? null : Bytes(expected), ideal, bound) };
            if (!layout.Direct) { outputs["partial"] = new(null, ScratchBytes: checked((int)layout.PartialBytes)); outputs["maxima"] = new(null, ScratchBytes: checked((int)layout.StatBytes)); outputs["denominators"] = new(null, ScratchBytes: checked((int)layout.StatBytes)); }
            yield return new($"swa-{(dense ? "bounded-dense" : "analytic-zero-qk")}-b{block}-h{history}",
                new() { ["query"] = Bytes(query), ["key"] = Bytes(key), ["value"] = Bytes(value), ["positions"] = Bytes(positions), ["valid"] = Bytes(valid), ["lanes"] = Bytes(physical), ["ringK"] = Bytes(ringK), ["ringV"] = Bytes(ringV) }, outputs,
                (m, p) => m.Swa(p["query"], p["key"], p["value"], p["positions"], p["valid"], p["lanes"], p["ringK"], p["ringV"], p.GetValueOrDefault("partial"), p.GetValueOrDefault("maxima"), p.GetValueOrDefault("denominators"), p["output"], maxContext, block, batch, capacity, IntPtr.Zero),
                new { oracle = dense ? "FP64 full softmax over exactly visible history/current tokens" : "Zero QK: all admitted softmax weights exactly1; per-split integer sums<=72, exact denominator; final original direct reciprocal or split division then BF16", exactBits = !dense,
                    bounds = dense ? "Scoped |Q|,|K|<1/8, |V|<=1 envelope: 1/64+(1/128)*abs(FP64ideal); no universal claim" : "2^-8*abs(ideal)+8*2^-24 plus exact-bit gate", historyWindow = 2047, currentBlock = "all valid K/V bidirectional", physicalSlots = physical, valid,
                    scratchContract = "guards only; inactive poison and legitimate negative-infinity m are allowed", splitCapacity = layout.Splits });
        }
    }
}
