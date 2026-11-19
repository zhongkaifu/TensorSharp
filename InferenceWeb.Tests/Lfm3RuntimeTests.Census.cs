// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The steps' census (StandardDecoderModel.Census, IGraphPlanCensus; eng/lfm3/Lfm3Probe batch) counts what batched decode steps
// move and launch: one run a step, its input leaves, the logits it reads back, the executor's nodes and kernels - on direct CUDA
// whether the step was traced, recorded or replayed.
using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed partial class Lfm3RuntimeTests
{
    /// <summary>Batched decode steps on the managed executor's graphs and on the pinned GGML backend's: each step is one run
    /// that uploads its inputs, reads back each sequence's last-row logits and nothing else, and reports the executor's nodes
    /// and kernels (more for three sequences than for one: each attends its own slot and convolves its own state).</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void BatchedDecodeStepsCountTheirRunsInputsResultsAndKernels(string lane)
        => AssertStepCensus(lane == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu, Fixture("dense"), recorder: false);

    /// <summary>Direct CUDA records an advanced decode step and replays the recording (F16 caches and 128-wide heads, so the
    /// sliding layers' bands are read on the device and every step can record): the three-sequence cell is measured from its
    /// first step, so it holds a traced run, the recording run and replays, and each counts as one run that reads back each
    /// sequence's logits once - the recording run's results included, which it keeps on the device until the step copies
    /// them out as a replay's.</summary>
    [CudaFact]
    public void DirectCudaRecordedDecodeStepsCountInTheCensus()
        => AssertStepCensus(BackendType.Cuda, Fixture("dense128"), recorder: true);

    private void AssertStepCensus(BackendType backend, string fixture, bool recorder)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using Lfm3RuntimeModel model = Create(fixture, backend: backend);
        var unified = (IUnifiedBatchModel)model;
        string[] ids = { "a", "b", "c" };
        var next = new int[ids.Length];
        var lengths = new int[ids.Length];
        for (int s = 0; s < ids.Length; s++)
        {
            int[] prompt = Enumerable.Range(0, 7 + 5 * s).Select(i => 30 + (i * 13 + s) % 200).ToArray();
            unified.EnsureSequenceSlot(ids[s], 128);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { ids[s] }, new[] { prompt }, new[] { 0 }, logits));
            next[s] = ArgMax(logits[0]);
            lengths[s] = prompt.Length;
        }

        (StandardDecoderModel.StepCensus Delta, int Recordings, int Replays) Decode(int sequences, int steps)
        {
            StandardDecoderModel.StepCensus before = model.Census.Clone();
            int runs = model.GraphRuns, recordings = model.DecodeRecordings, replays = model.RecordedDecodeReplays;
            for (int t = 0; t < steps; t++)
            {
                var logits = new float[sequences][];
                Assert.True(unified.TryForwardUnified(ids[..sequences], Enumerable.Range(0, sequences).Select(s => new[] { next[s] }).ToArray(),
                    lengths[..sequences], logits));
                for (int s = 0; s < sequences; s++)
                {
                    next[s] = ArgMax(logits[s]);
                    lengths[s]++;
                }
            }
            StandardDecoderModel.StepCensus after = model.Census;
            Assert.Equal(steps, model.GraphRuns - runs);   // every step ran as a whole-forward graph
            return (new StandardDecoderModel.StepCensus
            {
                Runs = after.Runs - before.Runs, InputLeaves = after.InputLeaves - before.InputLeaves,
                InputBytes = after.InputBytes - before.InputBytes, OutputBytes = after.OutputBytes - before.OutputBytes,
                RunTicks = after.RunTicks - before.RunTicks, CensusRuns = after.CensusRuns - before.CensusRuns,
                Nodes = after.Nodes - before.Nodes, Kernels = after.Kernels - before.Kernels,
                LeafUploads = after.LeafUploads - before.LeafUploads, LeafUploadBytes = after.LeafUploadBytes - before.LeafUploadBytes,
                LeafDownloads = after.LeafDownloads - before.LeafDownloads, LeafDownloadBytes = after.LeafDownloadBytes - before.LeafDownloadBytes,
            }, model.DecodeRecordings - recordings, model.RecordedDecodeReplays - replays);
        }

        const int Steps = 6;
        // Warm: each shape's first steps build its graph (and a recorder's first run records it). With a recorder the
        // three-sequence cell is measured cold instead: its trace, its recording and its replays are all counted.
        Decode(1, 2);
        if (!recorder) Decode(3, 2);
        (StandardDecoderModel.StepCensus one, int oneRecordings, int oneReplays) = Decode(1, Steps);
        (StandardDecoderModel.StepCensus three, int threeRecordings, int threeReplays) = Decode(3, Steps);
        _output.WriteLine($"{backend}: one sequence {one.InputLeaves / Steps} inputs, {one.InputBytes / Steps} B, {one.Kernels / Steps} kernels of " +
            $"{one.Nodes / Steps} nodes, {one.LeafUploads / Steps} leaves uploaded, {oneRecordings} recordings, {oneReplays} replays; three " +
            $"{three.InputLeaves / Steps} inputs, {three.InputBytes / Steps} B, {three.Kernels / Steps} kernels of {three.Nodes / Steps} nodes, " +
            $"{three.LeafUploads / Steps} leaves uploaded, {threeRecordings} recordings, {threeReplays} replays");
        if (recorder)
        {
            // The census hooks of TryRunRecorded ran: the cold cell recorded and replayed after it, the warm cell replayed.
            Assert.True(threeRecordings >= 1, $"{threeRecordings} recordings of the three sequences' step");
            Assert.True(threeReplays >= 1, $"{threeReplays} replays of the three sequences' recording");
            Assert.True(oneReplays >= 1, $"{oneReplays} replays of the lone sequence's recording");
        }
        foreach ((StandardDecoderModel.StepCensus c, int sequences) in new[] { (one, 1), (three, 3) })
        {
            Assert.Equal(Steps, c.Runs);
            Assert.Equal(Steps, c.CensusRuns);
            Assert.True(c.InputLeaves >= Steps && c.InputBytes > 0, $"{c.InputLeaves} input leaves of {c.InputBytes} B");
            Assert.Equal((long)Steps * sequences * Lfm3SyntheticModelBuilder.Vocab * sizeof(float), c.OutputBytes);
            Assert.True(c.Kernels > 0 && c.Nodes >= c.Kernels, $"{c.Kernels} kernels of {c.Nodes} nodes");
            Assert.True(c.RunTicks > 0);
        }
        Assert.True(three.Kernels > one.Kernels, $"three sequences {three.Kernels / Steps} kernels a step, one {one.Kernels / Steps}");
        Assert.True(three.InputLeaves >= one.InputLeaves);
        foreach (string id in ids) model.OnSequenceReleased(id);
    }
}
