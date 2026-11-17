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
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Decode rows and verify windows on direct CUDA attend their slot's whole capacity (IGraphExecutor.AttendsCapacityWindows),
/// so a request records each width once: a 300-token prompt's decode over 500 positions (across the 512, 768 and 1,024 key
/// buckets) records one step where bucket windows recorded one a bucket, and 4-row verify windows over the next 400 fewer than
/// bucket windows did - with the logits of bucket windows, bit for bit (recorded steps replayed at every length), over Qwen 3.5's
/// dense layout and Qwen3.8 Flash Next's (QSA selecting past its budget of 11 cells).
/// </summary>
public sealed class CapacityWindowRecordingTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("capacity-windows").FullName;
    private readonly ITestOutputHelper _output;

    public CapacityWindowRecordingTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private sealed record Trace(List<float[]> Rows, int DecodeRecordings, int WindowRecordings, int Builds, int Replays);

    [CudaTheory]
    [InlineData("qwen35")]
    [InlineData("qwen4exp")]
    public void ARequestRecordsEachWidthOnceAndAnswersAsBucketWindows(string family)
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = family == "qwen35"
            ? new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"))
            : Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 8, qsa: true);
        Trace capacity = Run(path, family, bucketWindows: false);
        Trace bucket = Run(path, family, bucketWindows: true);
        _output.WriteLine($"capacity windows: {capacity.DecodeRecordings} + {capacity.WindowRecordings} recordings, {capacity.Builds} builds; " +
            $"bucket windows: {bucket.DecodeRecordings} + {bucket.WindowRecordings} recordings, {bucket.Builds} builds");
        Assert.True(capacity.Replays > 0 && bucket.Replays > 0, "the steps did not replay");
        Assert.Equal(1, capacity.DecodeRecordings);
        Assert.Equal(3, bucket.DecodeRecordings);
        Assert.True(capacity.WindowRecordings < bucket.WindowRecordings,
            $"the windows recorded {capacity.WindowRecordings} times over the capacity, {bucket.WindowRecordings} over buckets");
        Assert.Equal(bucket.Rows.Count, capacity.Rows.Count);
        for (int i = 0; i < bucket.Rows.Count; i++)
            Assert.True(bucket.Rows[i].AsSpan().SequenceEqual(capacity.Rows[i]), $"step {i} differs over the capacity window");
    }

    /// <summary>A 300-token prompt, 500 decode rows (lengths 301 to 800), then 100 verify windows of 4 rows (to 1,200), in a
    /// slot of at least 2,048 cells: its logits and the recordings each phase made.</summary>
    private static Trace Run(string path, string family, bool bucketWindows)
    {
        using StandardDecoderModel model = family == "qwen35" ? new Qwen35RuntimeModel(path, BackendType.Cuda) : new Qwen4ExpRuntimeModel(path, BackendType.Cuda);
        model.CapacityWindowsDisabledForTests = bucketWindows;
        model.ResetKVCache();
        var spec = (ISpeculativeTarget)model;
        spec.SpecEnsureCapacity(2048);
        int vocab = model.Config.VocabSize;
        int[] Tokens(int from, int n) => Enumerable.Range(from, n).Select(i => 3 + (i * 37) % 250).ToArray();
        var rows = new List<float[]> { (float[])model.ForwardRefill(Tokens(0, 300)).Clone() };
        for (int i = 0; i < 500; i++)
            rows.Add((float[])model.Forward(Tokens(300 + i, 1)).Clone());
        int decodeRecordings = model.DecodeRecordings;
        int[] accepted = { 3, 0, 2, 1 };
        for (int w = 0; w < 100; w++)
        {
            var logits = new float[4 * vocab];
            spec.SpecForward(Tokens(800 + 4 * w, 4), null, logits, true);
            spec.SpecOnVerifyAccepted(accepted[w % accepted.Length], 4);
            rows.Add(logits);
        }
        return new Trace(rows, decodeRecordings, model.DecodeRecordings - decodeRecordings, model.GraphBuilds, model.GraphReplays);
    }
}
