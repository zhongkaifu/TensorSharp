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
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen3.8-Flash-Next (qwen4exp) on the shared decoder runtime: the mechanism tests every family on the runtime declares
/// (<see cref="SharedRuntimeConformanceTests.MechanismTests"/>), on the synthetic checkpoint (hyper-connections, the PLE
/// n-gram layer, QSA, Gated DeltaNet and a shared-expert MoE). The family runs only as graphs, so there is no per-op path to
/// compare them with (<see cref="SharedRuntimeConformanceTests.GraphOnlyFamilies"/>). Devices are simulated
/// (SimulatedDeviceGroup); the GPU runs are Qwen4ExpRuntimeTests' and Qwen4ExpCudaPlacementTests'.
/// </summary>
[Collection("MoeCpuOffloadConfig")]   // the host-experts test sets the process-wide expert-offload policy
public sealed class Qwen4ExpRuntimeConformanceTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("q4e-conformance").FullName;

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>The synthetic checkpoint with BF16 projections and experts: a Q8_0 product quantizes its activations, and a
    /// rank's repacked in-projections round apart from one device's by that noise (Qwen4ExpRuntimeTests' rank tests).</summary>
    private string Fixture(bool qsa = true) => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, $"q4e-{qsa}.gguf"), indexerTopK: 64,
        qsa: qsa, projectionType: SyntheticGguf.GgmlType.BF16, expertGateUpType: SyntheticGguf.GgmlType.BF16,
        expertDownType: SyntheticGguf.GgmlType.BF16);

    private static Qwen4ExpRuntimeModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1, int stages = 1)
        => new(path, backend, layerSplitDegree: stages, tpDegree: tp, tpGroup: tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null);

    /// <summary>Tensor parallelism (per-rank graphs: attention and Gated DeltaNet heads and every expert's width split, the
    /// hyper-connection mixers, the PLE layer and the QSA indexer whole on every rank), whole-layer stages (the hyper-connection
    /// streams and the per-layer embeddings passing on) and their hybrid answer as one device.</summary>
    [Theory]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 2)]
    public void PlacementMatchesOneDevice(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        foreach (bool qsa in new[] { false, true })
        {
            string path = Fixture(qsa);
            float[][] one = Script(() => Create(path));
            float[][] placed = Script(() =>
            {
                Qwen4ExpRuntimeModel model = Create(path, BackendType.Cpu, tp, stages);
                Assert.Equal(tp, model.Mesh.TensorParallel);
                Assert.Equal(stages, model.Mesh.Stages);
                Assert.True(tp == 1 || model.RunsRankGraphs, $"tp={tp} stages={stages}: the placement does not run per-rank graphs");
                return model;
            });
            // Whole-layer stages run the very same ops on other allocators; the ranks' partials meet in another order.
            Qwen4ExpRuntimeTests.AssertClose(one, placed, tp == 1 ? 1e-6 : 1e-4, $"tp={tp} stages={stages} qsa={qsa}");
        }
    }

    /// <summary>Batched slots (prompt chunks and decode rows of several sequences in one step) give each sequence the rows it
    /// gets alone, in every placement.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 2)]
    public void SlotBatchesMatchEachSequenceAlone(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, BackendType.Cpu, tp, stages), tp == 1 ? 1e-5 : 1e-4,
            $"qwen4exp tp={tp} stages={stages}");
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta, the prefix cache) equal each request's alone, on one device and over the hybrid.</summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 2)]
    public async Task ConcurrentRequestsAreBatched(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 41).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        var expected = new List<int[]>();
        using (Qwen4ExpRuntimeModel reference = Create(path, BackendType.Cpu, tp, stages))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = Argmax(logits);
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    produced.Add(token);
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using Qwen4ExpRuntimeModel model = Create(path, BackendType.Cpu, tp, stages);
        Assert.True(((IBatchedPagedModel)model).SupportsPerSequenceFusedForward);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            NumBlocks = 32,
            BlockSize = 16,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"req{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));

        Assert.True(model.BatchedForwardCount > 0, "the engine must serve these requests in batched steps over their slots");
        for (int i = 0; i < prompts.Count; i++)
            Assert.Equal(expected[i], outputs[i]);
    }

    /// <summary>Routed experts kept in host memory (--cpu-moe) answer as the device's: the tensor executor's expert host on
    /// the managed backend, on one device and over whole-layer stages, the shared expert staying in the graph.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void HostExpertsMatchDeviceExperts(int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        MoeCpuOffloadConfig.Reset();
        float[][] device = Script(() => Create(path));
        try
        {
            MoeCpuOffloadConfig.SetAllLayers();
            float[][] host = Script(() => Create(path, BackendType.Cpu, 1, stages));
            Qwen4ExpRuntimeTests.AssertClose(device, host, 1e-5, $"--cpu-moe, {stages} stage(s)");
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>A shared-prefix checkpoint, cloned or written to a file and read back, continues exactly: the recurrent
    /// states, the PLE layer's n-gram and convolution histories and the K/V rows together.</summary>
    [Fact]
    public void CheckpointsContinueExactly()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path), 1e-6, "qwen4exp managed");
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, TestGates.PinnedGgmlBackend), 1e-5, "qwen4exp ggml",
            deviceCopies: true);
    }

    /// <summary>Greedy speculation (n-gram drafts, and the shared MTP head) emits what plain greedy decoding emits.</summary>
    [Fact]
    public void SpeculativeDecodingMatchesPlainGreedy()
    {
        using var suite = new Qwen4ExpRuntimeTests();
        suite.SpeculationMatchesPlainGreedyOnTheManagedCpu();
    }

    /// <summary>Prompt, then decode rows, the logits of each step copied (a model may reuse its logits buffer).</summary>
    private static float[][] Script(Func<ModelBase> create)
    {
        int[] prompt = { 1, 17, 33, 2, 99, 5, 255, 45, 12, 200, 7, 63, 88, 31 };
        int[] decode = { 41, 9, 130, 77 };
        using ModelBase model = create();
        model.ResetKVCache();
        model.ForwardRefill(prompt[..6]);
        var rows = new List<float[]> { (float[])model.Forward(prompt[6..]).Clone() };
        foreach (int token in decode)
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        return rows.ToArray();
    }

    private static int Argmax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
}
