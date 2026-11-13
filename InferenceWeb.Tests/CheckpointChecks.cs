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
    /// <param name="deviceCopies">The model runs on a GGML backend: the checks run once as loaded, once with every
    /// slot copy made by the device copy program, which must then have been used, and once with slots that served a
    /// request waiting in the pool (the checkpoint is copied into one on the device, its buffers already bound).</param>
    public static void AssertCheckpointsContinueExactly(Func<ModelBase> create, double tolerance, string label, bool deviceCopies = false)
    {
        int[] prefix = Enumerable.Range(0, 23).Select(i => 40 + (i * 7) % 60).ToArray();
        int[] tail = { 71, 72, 73 };

        var expected = new List<float[]>();
        using (ModelBase reference = create())
        {
            reference.ForwardRefill(prefix);
            foreach (int t in tail) expected.Add((float[])reference.Forward(new[] { t }).Clone());
        }

        foreach (string mode in deviceCopies ? new[] { "", "device copies", "pooled slots" } : new[] { "" })
        {
            string run = mode.Length > 0 ? $"{label} {mode}" : label;
            byte[] file;
            using (ModelBase model = create())
            {
                var slots = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
                Assert.True(slots.SupportsPrefixCheckpoints, $"{run}: no checkpoints");
                var runtime = model as StandardDecoderModel;
                if (mode == "device copies") runtime!.ForceDeviceSlotCopies = true;
                int copiesBefore = runtime?.DeviceSlotCopies ?? 0;
                if (mode == "pooled slots")
                {
                    // Two slots run a request each and wait in the pool: "a" takes one, the checkpoint the other.
                    foreach (string id in new[] { "w1", "w2" })
                    {
                        slots.BindSequenceCache(id);
                        model.ForwardRefill(new[] { 11, 12, 13 });
                    }
                    slots.RestorePrimaryCache();
                    slots.OnSequenceReleased("w1");
                    slots.OnSequenceReleased("w2");
                }
                slots.BindSequenceCache("a");
                model.ForwardRefill(prefix);
                Assert.True(slots.TryCheckpointActiveCache("shared"), $"{run}: checkpoint refused");
                model.Forward(new[] { 99 });   // the source request moves on; the checkpoint must not

                Assert.True(slots.TryCloneRetainedCache("shared", "b"));
                Assert.False(slots.BindSequenceCache("b"), "the clone is the request's slot, not a fresh one");
                Compare(model, tail, expected, tolerance, $"{run} clone");
                // Where graphs left the rows on the device, the checkpoint and the clone never went through the host.
                if (runtime != null && runtime.GraphRuns > 0 && runtime.SlotCopiesStayOnDevice)
                    Assert.True(runtime.DeviceSlotCopies >= 2, $"{run}: {runtime.DeviceSlotCopies} slot copies on the device");
                if (mode == "pooled slots" && runtime != null && runtime.GraphRuns > 0)
                    Assert.True(runtime.DeviceSlotCopies > copiesBefore, $"{run}: the checkpoint into a bound pooled slot went through the host");

                using var stream = new MemoryStream();
                Assert.True(slots.TryExportRetainedCache("shared", stream));
                file = stream.ToArray();
                // Writing the file synced the checkpoint to the host and left its device copies as they were: a clone
                // still copies on the device (a server exports every shared-prefix checkpoint it keeps).
                if (runtime != null && runtime.GraphRuns > 0 && runtime.SlotCopiesStayOnDevice && runtime.KeepsDeviceCopies)
                {
                    int beforeClone = runtime.DeviceSlotCopies;
                    Assert.True(slots.TryCloneRetainedCache("shared", "d"));
                    Assert.True(runtime.DeviceSlotCopies > beforeClone, $"{run}: a clone of an exported checkpoint went through the host");
                    slots.BindSequenceCache("d");
                    Compare(model, tail, expected, tolerance, $"{run} clone after export");
                }
            }

            using ModelBase other = create();
            var otherSlots = (IBatchedPagedModel)other;
            Assert.True(otherSlots.TryImportRetainedCache("restored", new MemoryStream(file)), $"{run}: import refused");
            Assert.True(otherSlots.TryCloneRetainedCache("restored", "c"));
            otherSlots.BindSequenceCache("c");
            Compare(other, tail, expected, tolerance, $"{run} file");
        }
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
