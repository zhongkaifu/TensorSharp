// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Batched decode steps on direct CUDA record and replay as a lone sequence's do: each sequence's attention and cache append
/// read its own length and write row on the device (named by its caches), over its slot's capacity window. Three requests at
/// different positions decode together in three slot sets - two of them, then all three (the third joins), then two again
/// (the second leaves) - crossing the 256 and 512 key buckets: one recording a slot set, and every row's logits those of the
/// same steps unrecorded, and of today's batched steps (unrecorded, over bucket windows), bit for bit; over Qwen 3.5's dense
/// layout and Qwen3.8 Flash Next's (QSA selecting past its budget of 11 cells). A request's single-sequence decode of the same
/// tokens is reported against its batched rows (the batched products may round apart from one-row ones). The recording's
/// parameter blocks are checked on their own too: each sequence's launches find its blocks by its caches.
/// </summary>
public sealed class BatchedDecodeRecordingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("batched-recordings").FullName;
    private readonly ITestOutputHelper _output;

    public BatchedDecodeRecordingTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly string[] Ids = { "a", "b", "c" };
    private static readonly int[] PromptLengths = { 200, 240, 30 };

    /// <summary>The slot sets in turn and their steps: a and b (across 256), all three, a and c (a across 512, c across 256).</summary>
    private static readonly (int[] Members, int Steps)[] Phases = { (new[] { 0, 1 }, 100), (new[] { 0, 1, 2 }, 60), (new[] { 0, 2 }, 200) };

    private sealed record Trace(List<float[]> Rows, List<float[]> FirstRequestRows, int Recordings, int Replays);

    [CudaTheory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    public void BatchedStepsRecordOnceASlotSetAndAnswerAsUnrecorded(string family)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = family == "qwen35"
            ? new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"))
            : Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, qsa: true);
        Trace recorded = Run(path, family, record: true, capacityWindows: true);
        Trace unrecorded = Run(path, family, record: false, capacityWindows: true);
        Trace today = Run(path, family, record: false, capacityWindows: false);
        _output.WriteLine($"{recorded.Recordings} recordings, {recorded.Replays} replays recorded; {unrecorded.Replays} advances unrecorded");
        Assert.Equal(Phases.Length, recorded.Recordings);
        Assert.Equal(0, unrecorded.Recordings);
        Assert.True(recorded.Replays > Phases.Sum(p => p.Steps) / 2, "the batched steps did not replay");
        AssertSameBits(unrecorded.Rows, recorded.Rows, "recorded");
        AssertSameBits(today.Rows, unrecorded.Rows, "over capacity windows");

        // The first request alone through the same tokens: reported, not required (row counts pick the products' kernels).
        List<float[]> alone = RunAlone(path, family);
        double worst = 0;
        int same = 0;
        for (int i = 0; i < alone.Count; i++)
        {
            if (alone[i].AsSpan().SequenceEqual(recorded.FirstRequestRows[i])) same++;
            worst = Math.Max(worst, RelL2(alone[i], recorded.FirstRequestRows[i]));
        }
        _output.WriteLine($"request a alone: {same} of {alone.Count} rows bit for bit its batched rows, worst relL2 {worst:E2}");
    }

    /// <summary>
    /// Two sequences' 2-row verify windows in one step, each keeping its own count every time (one, then two, alternately and
    /// out of phase): the next window advances each sequence by its own rows and replays the step's recording, where one count
    /// for all had traced every such window again. Every row's logits are those of the same windows traced (no advance, no
    /// recording), bit for bit; over Qwen 3.5's dense layout and Qwen3.8 Flash Next's.
    /// </summary>
    [CudaTheory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    public void BatchedVerifyWindowsKeepingDifferentRowsReplayTheirRecording(string family)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = family == "qwen35"
            ? new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense-w.gguf"))
            : Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-w.gguf"), indexerTopK: 8, qsa: true);
        (List<float[]> replayed, int advances, int recordings) = RunWindows(path, family, advance: true);
        (List<float[]> traced, int tracedAdvances, _) = RunWindows(path, family, advance: false);
        _output.WriteLine($"{advances} window advances, {recordings} recordings; traced {tracedAdvances}");
        Assert.Equal(0, tracedAdvances);
        Assert.True(advances >= WindowSteps - 2, $"{advances} of {WindowSteps} batched windows advanced");
        Assert.True(recordings >= 1, "the batched windows never recorded");
        AssertSameBits(traced, replayed, "advanced by each sequence's rows");
    }

    private const int WindowSteps = 40;

    /// <summary>Requests a and b's prompts, then <see cref="WindowSteps"/> batched 2-row verify windows: every row's logits,
    /// the windows that advanced and the recordings made.</summary>
    private static (List<float[]> Rows, int Advances, int Recordings) RunWindows(string path, string family, bool advance)
    {
        using Qwen35RuntimeModel model = Create(path, family);
        model.DecodeAdvanceDisabled = !advance;
        model.DecodeCaptureDisabled = !advance;
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var batched = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.IBatchedSpeculativeTarget>(model);
        string[] ids = { Ids[0], Ids[1] };
        var starts = new int[2];
        for (int r = 0; r < 2; r++)
        {
            unified.EnsureSequenceSlot(ids[r], 1024);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { ids[r] }, new[] { Prompt(r) }, new[] { 0 }, logits));
            starts[r] = PromptLengths[r];
        }
        const int window = 2;
        int vocab = model.Config.VocabSize, advances = model.WindowAdvances, recordings = model.DecodeRecordings;
        var rows = new List<float[]>();
        var all = new float[2 * window * vocab];
        for (int step = 0; step < WindowSteps; step++)
        {
            var tokens = new int[2 * window];
            for (int r = 0; r < 2; r++)
                for (int t = 0; t < window; t++) tokens[r * window + t] = Token(r, step * window + t);
            int[] at = (int[])starts.Clone();
            Assert.True(batched.TryBatchedSpecVerifyLogits(ids, at, window, tokens, all, null));
            for (int row = 0; row < 2 * window; row++) rows.Add(all.AsSpan(row * vocab, vocab).ToArray());
            // a keeps one row, then two; b two, then one: the sequences move on by different rows at every step.
            int[] kept = { 1 + step % 2, 2 - step % 2 };
            batched.CommitBatchedSpecVerify(ids, at, kept);
            for (int r = 0; r < 2; r++) starts[r] += kept[r];
        }
        return (rows, model.WindowAdvances - advances, model.DecodeRecordings - recordings);
    }

    /// <summary>A step of more sequences than the 16 a recording had held (24 here, a third of them joining late) records once
    /// a slot set and gives the unrecorded steps' bits: the blocks past the 16th are each sequence's own, and a replay uploads
    /// as many as it records.</summary>
    [CudaFact]
    public void StepsOfMoreThanSixteenSequencesRecordAndAnswerAsUnrecorded()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense-wide.gguf"));
        (List<float[]> Rows, int Recordings, int Replays) Wide(bool record)
        {
            using Qwen35RuntimeModel model = Create(path, "qwen35");
            model.DecodeCaptureDisabled = !record;
            var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
            const int n = 24, early = 16, steps = 12;
            string[] ids = Enumerable.Range(0, n).Select(i => $"w{i}").ToArray();
            var starts = new int[n];
            for (int r = 0; r < n; r++)
            {
                unified.EnsureSequenceSlot(ids[r], 512);
                int[] prompt = Enumerable.Range(0, 20 + 7 * r).Select(i => 5 + (i * 13 + r * 7) % 240).ToArray();
                var one = new float[1][];
                Assert.True(unified.TryForwardUnified(new[] { ids[r] }, new[] { prompt }, new[] { 0 }, one));
                starts[r] = prompt.Length;
            }
            int recordings = model.DecodeRecordings, replays = model.RecordedDecodeReplays;
            var rows = new List<float[]>();
            foreach (int members in new[] { early, n })
                for (int step = 0; step < steps; step++)
                {
                    var at = new int[members];
                    var tokens = new int[members][];
                    for (int r = 0; r < members; r++) { at[r] = starts[r]; tokens[r] = new[] { Token(r, step) }; }
                    var logits = new float[members][];
                    Assert.True(unified.TryForwardUnified(ids.Take(members).ToArray(), tokens, at, logits));
                    for (int r = 0; r < members; r++) { rows.Add((float[])logits[r].Clone()); starts[r]++; }
                }
            return (rows, model.DecodeRecordings - recordings, model.RecordedDecodeReplays - replays);
        }
        var recorded = Wide(record: true);
        var unrecorded = Wide(record: false);
        _output.WriteLine($"{recorded.Recordings} recordings, {recorded.Replays} replays recorded; {unrecorded.Recordings} unrecorded");
        Assert.Equal(2, recorded.Recordings);   // the 16-sequence set and the 24-sequence set
        Assert.Equal(0, unrecorded.Recordings);
        Assert.True(recorded.Replays >= 12, $"{recorded.Replays} recorded replays: the steps did not replay");
        AssertSameBits(unrecorded.Rows, recorded.Rows, "recorded past 16 sequences");
    }

    /// <summary>A batched recording's launches find their sequence's blocks by the caches they name: each sequence's run of
    /// blocks (its verify rows' after its own), and a launch naming no sequence aborts the recording rather than read
    /// another's length.</summary>
    [CudaFact]
    public void EachSequenceReadsItsOwnBlocks()
    {
        using var allocator = new TensorSharp.Cuda.CudaAllocator();
        using var dyn = new TensorSharp.Cuda.CudaDecodeDynParams(allocator);
        Assert.True(dyn.IsValid);
        using var a = new Tensor(allocator, DType.Float16, 4);
        using var b = new Tensor(allocator, DType.Float16, 4);
        using var stranger = new Tensor(allocator, DType.Float16, 4);
        dyn.Write(new[] { 300, 41 }, new[] { 299, 40 });
        dyn.NameSequences(new IReadOnlyList<Tensor>[] { new[] { a }, new[] { b } });
        dyn.Activate();
        try
        {
            const int block = 4 * sizeof(int), perSequence = (1 + TensorSharp.Cuda.CudaDecodeDynParams.WindowRowBlocks) * block;
            Assert.Equal(dyn.DevicePtr, TensorSharp.Cuda.CudaDecodeDynParams.GetActiveDevicePtr(allocator, a.Storage));
            Assert.Equal(dyn.DevicePtr + perSequence, TensorSharp.Cuda.CudaDecodeDynParams.GetActiveDevicePtr(allocator, b.Storage));
            TensorSharp.Cuda.CudaDecodeDynParams.WindowRow(2);
            Assert.Equal(dyn.DevicePtr + perSequence + 3 * block, TensorSharp.Cuda.CudaDecodeDynParams.GetActiveDevicePtr(allocator, b.Storage));
            TensorSharp.Cuda.CudaDecodeDynParams.WindowRow(null);
            Assert.Throws<TensorSharp.Cuda.CudaGraphCaptureAbortedException>(
                () => TensorSharp.Cuda.CudaDecodeDynParams.GetActiveDevicePtr(allocator, stranger.Storage));
            Assert.Throws<TensorSharp.Cuda.CudaGraphCaptureAbortedException>(() => TensorSharp.Cuda.CudaDecodeDynParams.GetActiveDevicePtr(allocator));
        }
        finally
        {
            TensorSharp.Cuda.CudaDecodeDynParams.Deactivate();
        }
    }

    private static Qwen35RuntimeModel Create(string path, string family)
        => family == "qwen35" ? new Qwen35RuntimeModel(path, BackendType.Cuda) : new Qwen4ExpRuntimeModel(path, BackendType.Cuda);

    private static int Token(int request, int step) => 3 + (step * 37 + request * 11) % 250;

    private static int[] Prompt(int request) => Enumerable.Range(0, PromptLengths[request]).Select(i => 5 + (i * 13 + request * 7) % 240).ToArray();

    /// <summary>The three requests' prompts, then the phases' batched decode steps: every row's logits, request a's rows apart,
    /// and the recordings and replays the steps made.</summary>
    private static Trace Run(string path, string family, bool record, bool capacityWindows)
    {
        using Qwen35RuntimeModel model = Create(path, family);
        model.DecodeCaptureDisabled = !record;
        model.CapacityWindowsDisabledForTests = !capacityWindows;
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var starts = new int[Ids.Length];
        var steps = new int[Ids.Length];
        for (int r = 0; r < Ids.Length; r++)
        {
            unified.EnsureSequenceSlot(Ids[r], 1024);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { Ids[r] }, new[] { Prompt(r) }, new[] { 0 }, logits));
            starts[r] = PromptLengths[r];
        }
        int recordings = model.DecodeRecordings, replays = model.GraphReplays;
        var rows = new List<float[]>();
        var first = new List<float[]>();
        foreach ((int[] members, int count) in Phases)
            for (int step = 0; step < count; step++)
            {
                string[] ids = members.Select(m => Ids[m]).ToArray();
                int[][] tokens = members.Select(m => new[] { Token(m, steps[m]) }).ToArray();
                int[] at = members.Select(m => starts[m]).ToArray();
                var logits = new float[members.Length][];
                Assert.True(unified.TryForwardUnified(ids, tokens, at, logits));
                for (int i = 0; i < members.Length; i++)
                {
                    rows.Add((float[])logits[i].Clone());
                    if (members[i] == 0) first.Add((float[])logits[i].Clone());
                    starts[members[i]]++;
                    steps[members[i]]++;
                }
            }
        return new Trace(rows, first, model.DecodeRecordings - recordings, model.GraphReplays - replays);
    }

    /// <summary>Request a alone through its prompt and the tokens it decodes in <see cref="Run"/>.</summary>
    private static List<float[]> RunAlone(string path, string family)
    {
        using Qwen35RuntimeModel model = Create(path, family);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        unified.EnsureSequenceSlot(Ids[0], 1024);
        var logits = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { Ids[0] }, new[] { Prompt(0) }, new[] { 0 }, logits));
        int decodeSteps = Phases.Where(p => p.Members.Contains(0)).Sum(p => p.Steps);
        var rows = new List<float[]>();
        for (int step = 0; step < decodeSteps; step++)
        {
            Assert.True(unified.TryForwardUnified(new[] { Ids[0] }, new[] { new[] { Token(0, step) } }, new[] { PromptLengths[0] + step }, logits));
            rows.Add((float[])logits[0].Clone());
        }
        return rows;
    }

    private static void AssertSameBits(List<float[]> expected, List<float[]> actual, string label)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.True(expected[i].AsSpan().SequenceEqual(actual[i]), $"row {i} differs {label} ({RelL2(expected[i], actual[i]):E2} relative L2)");
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)a[i] * a[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
