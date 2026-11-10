// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

/// <summary>
/// The slot contract every model that serves requests from per-request slots must keep: several
/// requests' slots in batched steps (prefill chunks of different lengths, then decode, with a lone
/// step on one slot in between) reproduce each request run alone on the primary cache. Batching
/// changes which rows share a forward, not what any row computes.
/// </summary>
internal static class SlotParity
{
    public static void AssertBatchedSlotsMatchAlone(Func<ModelBase> create, double tolerance, string label)
    {
        int[][] prompts =
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 6).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        int[][] decode = { new[] { 11, 12, 13, 14 }, new[] { 21, 22, 23, 24 }, new[] { 31, 32, 33, 34 } };

        // Alone: each request on the primary cache, prefill then one token at a time.
        var alone = new List<float[]>[prompts.Length];
        using (ModelBase reference = create())
        {
            for (int s = 0; s < prompts.Length; s++)
            {
                reference.ResetKVCache();
                alone[s] = new List<float[]> { (float[])reference.Forward(prompts[s]).Clone() };
                foreach (int token in decode[s])
                    alone[s].Add((float[])reference.Forward(new[] { token }).Clone());
            }
        }

        using ModelBase model = create();
        var slots = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.True(slots.SupportsPerSequenceFusedForward);
        string[] ids = { "a", "b", "c" };
        foreach (string id in ids)
            Assert.True(slots.BindSequenceCache(id), "a new request gets a fresh slot");

        var batched = new List<float[]>[prompts.Length];
        for (int s = 0; s < prompts.Length; s++) batched[s] = new List<float[]>();
        var outLogits = new float[ids.Length][];
        Assert.True(slots.TryForwardBatchedFusedPrefill(ids, prompts, new int[ids.Length], outLogits));
        for (int s = 0; s < ids.Length; s++) batched[s].Add((float[])outLogits[s].Clone());

        for (int step = 0; step < decode[0].Length; step++)
        {
            if (step == 2)
            {
                // A lone step on request b's slot through the single-sequence forward.
                Assert.False(slots.BindSequenceCache("b"));
                batched[1].Add((float[])model.Forward(new[] { decode[1][step] }).Clone());
                var pairLogits = new float[2][];
                Assert.True(slots.TryForwardBatchedFusedDecode(new[] { "a", "c" },
                    new[] { decode[0][step], decode[2][step] },
                    new[] { prompts[0].Length + step, prompts[2].Length + step }, pairLogits));
                batched[0].Add((float[])pairLogits[0].Clone());
                batched[2].Add((float[])pairLogits[1].Clone());
                continue;
            }
            var tokens = new int[ids.Length];
            var positions = new int[ids.Length];
            for (int s = 0; s < ids.Length; s++)
            {
                tokens[s] = decode[s][step];
                positions[s] = prompts[s].Length + step;
                Assert.True(slots.CanBatchDecode(ids[s], positions[s]));
            }
            Assert.True(slots.TryForwardBatchedFusedDecode(ids, tokens, positions, outLogits));
            for (int s = 0; s < ids.Length; s++) batched[s].Add((float[])outLogits[s].Clone());
        }
        Assert.False(slots.TryForwardBatchedFusedDecode(new[] { "a" }, new[] { 1 }, new[] { 0 }, new float[1][]),
            "a request that does not continue its slot at that position is declined, not run");

        for (int s = 0; s < prompts.Length; s++)
            for (int i = 0; i < alone[s].Count; i++)
            {
                double rel = RelL2(batched[s][i], alone[s][i]);
                Assert.True(rel <= tolerance, $"{label} request {s} row {i}: relL2 {rel:E2}");
            }

        foreach (string id in ids) slots.OnSequenceReleased(id);
        Assert.Equal(0, ((IPrefixCacheModelDiagnostics)model).PrivateHolderCount);
    }

    private static double RelL2(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
