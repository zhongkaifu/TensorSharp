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
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Nemotron-H on the shared decoder runtime (<see cref="NemotronRuntimeModel"/>) over the synthetic layouts
/// (<see cref="NemotronSyntheticModelBuilder"/>: Mamba-2, attention without positions, ReLU^2 dense and routed experts, a
/// latent bottleneck, sidecar scales, per-layer expert counts): the legacy model's frozen logits, the pinned GGML backend and
/// the direct CUDA executor against the managed one, and every mechanism the runtime claims for the family (placements,
/// batched slots, continuous batching, host-served experts, checkpoints, speculation through n-gram drafts and through an MTP
/// head).
/// </summary>
[Collection("MoeCpuOffloadConfig")]   // the host-experts test sets the process-wide expert-offload policy
public sealed class NemotronRuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-nemotron-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public NemotronRuntimeTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Fixture(string name)
    {
        string path = Path.Combine(_dir, name + ".gguf");
        return File.Exists(path) ? path : NemotronLegacySynthetic.Get(name).Write(path);
    }

    private static NemotronRuntimeModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1, int stages = 1,
        string draft = null)
    {
        ITensorParallelGroup group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        return new NemotronRuntimeModel(path, backend, tp, group, stages, draft);
    }

    // ---------------------------------------------------------------- the forward

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        using ModelBase model = ModelBase.Create(Fixture("moe"), BackendType.Cpu);
        var runtime = Assert.IsType<NemotronRuntimeModel>(model);
        Assert.Equal(5, model.Config.NumLayers);
        Assert.Equal(NemotronRuntimeModel.Block.Mamba2, runtime.BlockOfLayerForTests(0));
        Assert.Equal(NemotronRuntimeModel.Block.FeedForward, runtime.BlockOfLayerForTests(1));
        Assert.Equal(NemotronRuntimeModel.Block.Attention, runtime.BlockOfLayerForTests(2));
    }

    [Fact]
    public void PerLayerExpertCountsAndTopKAreEachLayersOwn()
    {
        using var model = Create(Fixture("moe-per-layer"));
        Assert.Equal((8, 3), model.ExpertLayoutForTests(1));
        Assert.Equal((6, 2), model.ExpertLayoutForTests(4));
        Assert.Equal((0, 0), model.ExpertLayoutForTests(2));
    }

    /// <summary>
    /// How far two F32 runs of these fixtures' random layers part when only their summation order differs (the runtime's
    /// graphs sum products, the scan's state rows and the experts' weighted rows in another order than the legacy per-op
    /// loops); a wrong recipe part (a conv tap, a gate, a norm group, a route) moves these logits by 0.1 to 1.
    /// </summary>
    private const double NoiseTolerance = 0.025;

    /// <summary>The legacy model's logits, frozen before it was deleted (<see cref="NemotronLegacySynthetic"/>): the same math
    /// on the managed backend, every case's prompt row and decode rows (the same top token wherever the frozen top two are
    /// half a logit apart).</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    [InlineData("moe-latent-scaled")]
    [InlineData("moe-per-layer")]
    public void MatchesTheFrozenLegacyLogits(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        List<float[]> frozen = NemotronLegacySynthetic.Rows(name, path);
        using var model = Create(path);
        List<float[]> actual = NemotronLegacySynthetic.Run(model, NemotronLegacySynthetic.Get(name));
        Assert.Equal(frozen.Count, actual.Count);
        for (int i = 0; i < frozen.Count; i++)
        {
            double rel = RelL2(actual[i], frozen[i]);
            _output.WriteLine($"{name} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= NoiseTolerance, $"{name} row {i}: relL2 {rel:E3} against the legacy model");
            if (Margin(frozen[i]) > 0.5f) Assert.Equal(Argmax(frozen[i]), Argmax(actual[i]));
        }
    }

    /// <summary>The pinned GGML backend's rows are the managed backend's (ggml_cpu's F32 products round apart; ggml_cuda's
    /// quantize the activations of its quantized products).</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    [InlineData("moe-latent-scaled")]
    public void PinnedGgmlBackendMatchesTheManagedBackend(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        NemotronLegacySynthetic.Case c = NemotronLegacySynthetic.Get(name);
        List<float[]> managed, ggml;
        using (var model = Create(path)) managed = NemotronLegacySynthetic.Run(model, c);
        using (var model = Create(path, TestGates.PinnedGgmlBackend)) ggml = NemotronLegacySynthetic.Run(model, c);
        double tolerance = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? NoiseTolerance : 2 * NoiseTolerance;
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(ggml[i], managed[i]);
            _output.WriteLine($"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= tolerance, $"{name} {TestGates.PinnedGgmlBackend} row {i}: relL2 {rel:E3} > {tolerance:E1}");
        }
    }

    /// <summary>The direct CUDA executor (its selective-scan kernel, recorded decode steps) against the managed backend, and
    /// its runs reproduce bit for bit.</summary>
    [CudaTheory]
    [InlineData("dense")]
    [InlineData("moe")]
    [InlineData("moe-latent-scaled")]
    public void DirectCudaMatchesTheManagedBackend(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        NemotronLegacySynthetic.Case c = NemotronLegacySynthetic.Get(name);
        List<float[]> managed, first, again;
        using (var model = Create(path)) managed = NemotronLegacySynthetic.Run(model, c);
        using (var model = Create(path, BackendType.Cuda))
        {
            first = NemotronLegacySynthetic.Run(model, c);
            again = NemotronLegacySynthetic.Run(model, c);
        }
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(first[i], managed[i]);
            _output.WriteLine($"{name} cuda row {i}: relL2 {rel:E3}");
            Assert.True(rel <= 2 * NoiseTolerance, $"{name} cuda row {i}: relL2 {rel:E3}");
            Assert.Equal(first[i], again[i]);
        }
    }

    /// <summary>A prompt in one chunk and in pieces (the Mamba-2 conv windows and states carried across them) gives the same
    /// logits.</summary>
    [Theory]
    [InlineData("dense", "managed")]
    [InlineData("dense", "ggml")]
    [InlineData("moe", "managed")]
    [InlineData("moe", "ggml")]
    public void ChunkedPrefillMatchesOneChunk(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        using var model = Create(Fixture(name), backend);
        int[] prompt = NemotronLegacySynthetic.Get(name).Prompt;
        float[] whole = (float[])model.ForwardRefill(prompt).Clone();
        model.ResetKVCache();
        float[] pieces = null;
        int cut1 = prompt.Length / 3, cut2 = cut1 + 1;
        foreach (int[] chunk in new[] { prompt[..cut1], prompt[cut1..cut2], prompt[cut2..] })
            pieces = model.Forward(chunk);
        double tolerance = backend is BackendType.Cpu ? 1e-5 : backend == BackendType.GgmlCpu ? 5e-4 : 3e-2;
        Assert.True(RelL2(pieces, whole) <= tolerance, $"{name} {backend} chunked: relL2 {RelL2(pieces, whole):E3}");
    }

    // ---------------------------------------------------------------- mechanisms

    public static TheoryData<string, int, int> Placements() => new()
    {
        { "dense", 2, 1 },
        { "dense", 1, 2 },
        { "dense", 1, 3 },
        { "moe", 2, 1 },
        { "moe", 1, 2 },
        { "moe-latent-scaled", 2, 1 },
        { "dense", 2, 2 },
        { "moe", 2, 2 },
    };

    /// <summary>Tensor parallelism (per-rank graphs over simulated devices: each rank owns a block of Mamba-2 groups and the
    /// heads that read them, a slice of every expert's width), whole-layer stages and their hybrid (every stage's layers as
    /// per-rank graphs over its own group) answer as one device.</summary>
    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(string name, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        NemotronLegacySynthetic.Case c = NemotronLegacySynthetic.Get(name);
        List<float[]> reference;
        using (var one = Create(path)) reference = Script(one, c);
        using var placed = Create(path, BackendType.Cpu, tp, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed, c);
        Assert.True(placed.GraphRuns > 0, $"{name} tp={tp} stages={stages}: no graph ran");
        double tolerance = tp == 1 ? 1e-6 : NoiseTolerance;
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            _output.WriteLine($"{name} tp={tp} stages={stages} row {i}: relL2 {rel:E3}");
            Assert.True(rel <= tolerance, $"{name} tp={tp} stages={stages} row {i}: relL2 {rel:E3} > {tolerance:E2}");
        }
    }

    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void SlotBatchesMatchEachSequenceAlone(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), 1e-5, $"nemotron {name}");
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta, the prefix cache) equal each request's alone.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public async Task ConcurrentRequestsAreBatched(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 41).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        var expected = new List<int[]>();
        using (var reference = Create(path))
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

        using var model = Create(path);
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

    [Theory]
    [InlineData("dense", "managed")]
    [InlineData("dense", "ggml")]
    [InlineData("moe", "managed")]
    public void CheckpointsContinueExactly(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"nemotron {name} {backend}", deviceCopies: executor == "ggml");
    }

    /// <summary>Greedy speculation (scripted drafts, accepted and rejected, so verify windows take the accepted prefix's
    /// Mamba-2 states and conv windows) emits exactly the plain greedy tokens.</summary>
    [Theory]
    [InlineData("dense", "managed")]
    [InlineData("moe", "managed")]
    [InlineData("moe", "ggml")]
    public void SpeculativeDecodingMatchesPlainGreedy(string name, string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend), $"nemotron {name} {backend}",
            tieTolerance: backend == BackendType.Cpu ? 0 : 1e-2);
    }

    /// <summary>Routed experts kept in host memory (--cpu-moe) answer as the device's: on the managed backend (where host memory
    /// is the device's) on one device and over whole-layer stages, and on the pinned GGML backend (GGML's host MoE segment: the
    /// one-projection relu(up)^2 experts on the host, each route's up scale before the square, the host kernel quantizing the
    /// activations to the weights' dot type). The latent bottleneck and the NVFP4-style scales go through the same seam.</summary>
    [Theory]
    [InlineData("moe", "managed", 1)]
    [InlineData("moe", "managed", 2)]
    [InlineData("moe-latent-scaled", "managed", 1)]
    [InlineData("moe-per-layer", "managed", 1)]
    [InlineData("moe", "ggml", 1)]
    [InlineData("moe-latent-scaled", "ggml", 1)]
    [InlineData("moe-per-layer", "ggml", 1)]
    public void HostExpertsMatchDeviceExperts(string name, string executor, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture(name);
        NemotronLegacySynthetic.Case c = NemotronLegacySynthetic.Get(name);
        List<float[]> reference;
        MoeCpuOffloadConfig.Reset();
        using (var device = Create(path, backend)) reference = Script(device, c);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            ModelBase.ValidateExpertOffloadRequest(NemotronArchitecture.Descriptor, backend, 1);
            using var offloaded = Create(path, backend, 1, stages);
            List<float[]> actual = Script(offloaded, c);
            Assert.True(offloaded.GraphRuns > 0, $"{name} {backend}: no graph ran");
            // The managed backend runs the same products in another order; GGML's host segment quantizes the activations
            // its device products may keep in F32.
            double tolerance = backend == BackendType.Cpu ? 1e-4 : backend == BackendType.GgmlCpu ? NoiseTolerance : 2 * NoiseTolerance;
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                _output.WriteLine($"--cpu-moe {name} {backend} {stages} stage(s) row {i}: relL2 {rel:E3}");
                Assert.True(rel <= tolerance, $"--cpu-moe {name} {backend}, {stages} stage(s), row {i}: relL2 {rel:E3} > {tolerance:E1}");
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>The same on the direct CUDA executor: its expert host computes every host route with the host kernel
    /// (relu(s up)^2 over the up stack) and a prompt chunk wider than the kernel streams the selected experts to the device.</summary>
    [CudaTheory]
    [InlineData("moe")]
    [InlineData("moe-latent-scaled")]
    [InlineData("moe-per-layer")]
    public void HostExpertsOnDirectCudaMatchDeviceExperts(string name)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(name);
        NemotronLegacySynthetic.Case c = NemotronLegacySynthetic.Get(name);
        List<float[]> reference;
        MoeCpuOffloadConfig.Reset();
        using (var device = Create(path, BackendType.Cuda)) reference = Script(device, c);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            using var offloaded = Create(path, BackendType.Cuda);
            List<float[]> actual = Script(offloaded, c);
            (long hostRoutes, long streamed) = offloaded.ExpertHostCountsForTests;
            _output.WriteLine($"{name}: {hostRoutes} host routes, {streamed} streamed steps");
            Assert.True(hostRoutes > 0 && streamed > 0, $"{name}: host routes {hostRoutes}, streamed steps {streamed}");
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                _output.WriteLine($"--cpu-moe {name} cuda row {i}: relL2 {rel:E3}");
                Assert.True(rel <= 2 * NoiseTolerance, $"--cpu-moe {name} cuda row {i}: relL2 {rel:E3}");
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>The offload is served on every backend, on one device and over stages, and refused with --tp: the per-rank
    /// graphs keep the experts on the devices.</summary>
    [Fact]
    public void ExpertOffloadIsRefusedUnderTensorParallelism()
    {
        MoeCpuOffloadConfig.Reset();
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            var tp = Assert.Throws<NotSupportedException>(() => ModelBase.ValidateExpertOffloadRequest(NemotronArchitecture.Descriptor, BackendType.GgmlCuda, 2));
            Assert.Contains("--tp", tp.Message, StringComparison.Ordinal);
            foreach (BackendType backend in Enum.GetValues<BackendType>())
                ModelBase.ValidateExpertOffloadRequest(NemotronArchitecture.Descriptor, backend, 1);
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>A prompt chunk, decode steps, a second chunk and more steps: every step's logits.</summary>
    private static List<float[]> Script(ModelBase model, NemotronLegacySynthetic.Case c)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardRefill(c.Prompt).Clone() };
        foreach (int t in c.Decode.Take(3)) rows.Add((float[])model.Forward(new[] { t }).Clone());
        rows.Add((float[])model.Forward(NemotronLegacySynthetic.Tokens(9, 4)).Clone());
        foreach (int t in c.Decode.Skip(3)) rows.Add((float[])model.Forward(new[] { t }).Clone());
        return rows;
    }

    /// <summary>The top logit's lead over the runner-up.</summary>
    private static float Margin(float[] x)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float v in x)
        {
            if (v > first) (first, second) = (v, first);
            else if (v > second) second = v;
        }
        return first - second;
    }

    private static int Argmax(float[] x)
    {
        int best = 0;
        for (int i = 1; i < x.Length; i++) if (x[i] > x[best]) best = i;
        return best;
    }

    private static double RelL2(float[] a, float[] b)
    {
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
