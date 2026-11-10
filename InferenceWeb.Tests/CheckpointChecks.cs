// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>The shared-prefix checkpoint contract of a model that serves requests from slots: a
/// checkpoint taken after a prompt, cloned into a new request (or written to a file and read back
/// by another instance), continues exactly as a fresh run of the same tokens.</summary>
internal static class CheckpointChecks
{
    public static void AssertCheckpointsContinueExactly(Func<ModelBase> create, double tolerance, string label)
    {
        int[] prefix = Enumerable.Range(0, 23).Select(i => 40 + (i * 7) % 60).ToArray();
        int[] tail = { 71, 72, 73 };

        var expected = new List<float[]>();
        using (ModelBase reference = create())
        {
            reference.ForwardRefill(prefix);
            foreach (int t in tail) expected.Add((float[])reference.Forward(new[] { t }).Clone());
        }

        byte[] file;
        using (ModelBase model = create())
        {
            var slots = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
            Assert.True(slots.SupportsPrefixCheckpoints, $"{label}: no checkpoints");
            slots.BindSequenceCache("a");
            model.ForwardRefill(prefix);
            Assert.True(slots.TryCheckpointActiveCache("shared"), $"{label}: checkpoint refused");
            model.Forward(new[] { 99 });   // the source request moves on; the checkpoint must not

            Assert.True(slots.TryCloneRetainedCache("shared", "b"));
            Assert.False(slots.BindSequenceCache("b"), "the clone is the request's slot, not a fresh one");
            Compare(model, tail, expected, tolerance, $"{label} clone");

            using var stream = new MemoryStream();
            Assert.True(slots.TryExportRetainedCache("shared", stream));
            file = stream.ToArray();
        }

        using ModelBase other = create();
        var otherSlots = (IBatchedPagedModel)other;
        Assert.True(otherSlots.TryImportRetainedCache("restored", new MemoryStream(file)), $"{label}: import refused");
        Assert.True(otherSlots.TryCloneRetainedCache("restored", "c"));
        otherSlots.BindSequenceCache("c");
        Compare(other, tail, expected, tolerance, $"{label} file");
    }

    private static void Compare(ModelBase model, int[] tail, List<float[]> expected, double tolerance, string label)
    {
        for (int i = 0; i < tail.Length; i++)
        {
            float[] actual = model.Forward(new[] { tail[i] });
            double num = 0, den = 0;
            for (int v = 0; v < actual.Length; v++)
            {
                double d = actual[v] - expected[i][v];
                num += d * d;
                den += (double)expected[i][v] * expected[i][v];
            }
            double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
            Assert.True(rel <= tolerance, $"{label} row {i}: relL2 {rel:E2}");
        }
    }
}
