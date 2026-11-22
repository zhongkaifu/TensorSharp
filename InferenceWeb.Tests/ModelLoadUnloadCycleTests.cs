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
using System.Threading;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Models of several families (lfm3, qwen35, mistral3, gemma4, nemotron, qwen4exp, museglimmer) loaded and disposed in one
/// process, on the pinned GGML backend. Loaded and disposed in turn,
/// every load gives the logits of its first: nothing a disposed model leaves behind (a device copy, a freed block handed
/// out again) reaches the next one. A model kept loaded while others are loaded, run and disposed beside it carries on
/// exactly: each next decode step's logits are an uninterrupted run's, bit for bit, so its device copies, the plans bound
/// to them and its KV cache survived every dispose (a dispose had cleared the process's device copies: on ggml_cuda the
/// kept model's next step then crashed the process in TSGgml_GraphRun).
/// </summary>
public sealed class ModelLoadUnloadCycleTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("load-unload").FullName;
    private readonly ITestOutputHelper _output;

    public ModelLoadUnloadCycleTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 11).Select(i => 30 + (i * 13) % 160).ToArray();
    private static readonly int[] Decode = { 41, 77, 102 };

    private Dictionary<string, string> Fixtures() => new()
    {
        ["lfm3"] = new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3.gguf")),
        ["qwen35"] = new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true }.Write(Path.Combine(_dir, "q35.gguf")),
        ["mistral3"] = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }.Write(Path.Combine(_dir, "mistral3.gguf")),
        ["gemma4"] = new Gemma4DenseSyntheticModelBuilder().Write(Path.Combine(_dir, "gemma4.gguf")),
        ["nemotron"] = new NemotronSyntheticModelBuilder().Write(Path.Combine(_dir, "nemotron.gguf")),
        ["qwen4exp"] = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 64, qsa: false),
        ["museglimmer"] = new MuseGlimmerSyntheticModelBuilder().Write(Path.Combine(_dir, "museglimmer.gguf")),
    };

    /// <summary>The prompt's logits and three decode steps' of a fresh cache.</summary>
    private static List<float[]> Rows(ModelBase model)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardPrompt(Prompt).Clone() };
        foreach (int t in Decode)
            rows.Add((float[])model.Forward(new[] { t }).Clone());
        return rows;
    }

    /// <summary>A fresh cache's prompt, then the decode steps of <paramref name="tokens"/>; after each,
    /// <paramref name="between"/> (null to run uninterrupted).</summary>
    private static List<float[]> DecodeSteps(ModelBase model, int[] tokens, Action<int> between)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardPrompt(Prompt).Clone() };
        for (int i = 0; i < tokens.Length; i++)
        {
            between?.Invoke(i);
            rows.Add((float[])model.Forward(new[] { tokens[i] }).Clone());
        }
        return rows;
    }

    private static List<float[]> LoadRunDispose(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        return Rows(model);
    }

    private static void AssertSame(List<float[]> want, List<float[]> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int r = 0; r < want.Count; r++)
            Assert.True(want[r].AsSpan().SequenceEqual(got[r]), $"{what}: row {r} differs from the first load's");
    }

    [Fact]
    public void FamiliesLoadedAndDisposedInTurnGiveTheLogitsOfTheirFirstLoad()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        var first = paths.ToDictionary(p => p.Key, p => LoadRunDispose(p.Value, backend));
        // Two more rounds, each in another order, so every family follows several others' disposes.
        string[] order = paths.Keys.ToArray();
        for (int round = 1; round <= 2; round++)
        {
            string[] turn = round == 1 ? order.Reverse().ToArray() : order.Skip(order.Length / 2).Concat(order.Take(order.Length / 2)).ToArray();
            foreach (string family in turn)
                AssertSame(first[family], LoadRunDispose(paths[family], backend), $"{family}, round {round} on {backend}");
        }
        _output.WriteLine($"{backend}: {order.Length} families, three loads each, every load bit for bit its first");
    }

    [Fact]
    public void AModelKeptLoadedCarriesOnExactlyWhileTheOthersComeAndGo()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        string[] others = paths.Keys.Where(f => f != "lfm3").ToArray();
        int[] tokens = Enumerable.Range(0, others.Length + 1).Select(i => 41 + 17 * i).ToArray();
        List<float[]> uninterrupted;
        using (ModelBase alone = ModelBase.Create(paths["lfm3"], backend))
            uninterrupted = DecodeSteps(alone, tokens, null);
        using ModelBase kept = ModelBase.Create(paths["lfm3"], backend);
        // Before decode step i (i >= 1), another family is loaded, run and disposed beside the kept model, mid-sequence.
        List<float[]> carried = DecodeSteps(kept, tokens, i =>
        {
            if (i > 0) LoadRunDispose(paths[others[i - 1]], backend);
        });
        AssertSame(uninterrupted, carried, $"lfm3 kept loaded on {backend}");
        _output.WriteLine($"{backend}: lfm3 kept loaded through {others.Length} other loads and disposes mid-sequence, " +
            "every step bit for bit an uninterrupted run's");
    }

    /// <summary>
    /// A model decoding on another thread as an engine worker does (each step under its <see cref="ModelBase.GpuComputeLock"/>,
    /// yielding to registered waiters before it takes the lock) while two other models are loaded, run and disposed beside
    /// it mid-sequence: every step is an uninterrupted run's, bit for bit, with no crash. The others load and run between
    /// its steps; their disposes run beside them. A dispose's release of what models share (the graph scratch and graph
    /// caches) ran while the disposing thread held the decoding model's step lock: it waited for the step in progress, so
    /// no step computed over the memory it freed.
    /// </summary>
    [Fact]
    public void AModelDecodingOnAnotherThreadCarriesOnExactlyWhileOthersAreDisposed()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        int[] tokens = Enumerable.Range(0, 24).Select(i => 41 + 7 * i).ToArray();
        List<float[]> uninterrupted;
        using (ModelBase alone = ModelBase.Create(paths["lfm3"], backend))
            uninterrupted = DecodeSteps(alone, tokens, null);

        using ModelBase kept = ModelBase.Create(paths["lfm3"], backend);
        var decoding = new ManualResetEventSlim();
        var carried = new List<float[]>();
        Exception failure = null;
        var worker = new Thread(() =>
        {
            try
            {
                lock (kept.GpuComputeLock)
                {
                    kept.ResetKVCache();
                    carried.Add((float[])kept.ForwardPrompt(Prompt).Clone());
                }
                for (int i = 0; i < tokens.Length; i++)
                {
                    if (i == 2) decoding.Set();
                    GpuLockWaiters.YieldToWaiters(kept.GpuComputeLock, TimeSpan.FromSeconds(5));
                    lock (kept.GpuComputeLock)
                        carried.Add((float[])kept.Forward(new[] { tokens[i] }).Clone());
                }
            }
            catch (Exception e)
            {
                failure = e;
            }
            finally
            {
                decoding.Set();
            }
        });
        worker.Start();
        Assert.True(decoding.Wait(TimeSpan.FromMinutes(2)), "the worker never reached its third decode step");
        // The others load and run under the worker's step lock, as a second engine would take turns with it (two models'
        // graphs computing at once is not what this tests); each is disposed without it, so its release must take it.
        string probePath = Path.Combine(_dir, "release-probe.gguf");
        SyntheticGguf.Write(probePath, [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }], []);
        ModelBase other;
        ReleaseProbe probe;
        lock (kept.GpuComputeLock)
        {
            other = ModelBase.Create(paths["qwen35"], backend);
            Rows(other);
        }
        other.Dispose();
        lock (kept.GpuComputeLock)
            probe = new ReleaseProbe(probePath, backend, kept);
        probe.Dispose();
        Assert.True(worker.Join(TimeSpan.FromMinutes(5)), "the worker did not finish");
        Assert.Null(failure);
        Assert.True(probe.Released, "the probe's dispose ran no cross-model release (no other live model?)");
        Assert.True(probe.HeldTheDecodingModelsLock, "the shared release ran without the decoding model's step lock");
        AssertSame(uninterrupted, carried, $"lfm3 decoding on another thread on {backend}");
        _output.WriteLine($"{backend}: lfm3 decoded {tokens.Length} steps on another thread while qwen35 and a probe were " +
            "disposed beside it, bit for bit an uninterrupted run's; the release held its step lock");
    }

    /// <summary>
    /// A model that is retiring (an unload drains its uses) may still be finishing a media encode under its step lock:
    /// another model's dispose waits for that lock before it frees what models share, rather than skipping a model that is
    /// on its way out. Only a model whose dispose has begun is skipped (its disposer may hold its lock).
    /// </summary>
    [Fact]
    public void AReleaseWaitsForARetiringModelThatStillHoldsItsLock()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        string probePath = Path.Combine(_dir, "release-probe.gguf");
        SyntheticGguf.Write(probePath, [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }], []);
        using var retiring = new ReleaseProbe(probePath, backend, null);
        var probe = new ReleaseProbe(probePath, backend, retiring);
        retiring.BeginRetirement();
        var holding = new ManualResetEventSlim();
        var encoder = new Thread(() =>
        {
            lock (retiring.GpuComputeLock)
            {
                holding.Set();
                // Until the dispose below waits for this lock, then some of its 20 ms tries longer: a release that skipped
                // a retiring model would have run by then without it.
                SpinWait.SpinUntil(() => GpuLockWaiters.Count(retiring.GpuComputeLock) > 0, TimeSpan.FromMinutes(1));
                Thread.Sleep(200);
            }
        });
        encoder.Start();
        Assert.True(holding.Wait(TimeSpan.FromMinutes(1)), "the encoder never took the retiring model's lock");
        probe.Dispose();
        Assert.True(encoder.Join(TimeSpan.FromMinutes(2)), "the encoder did not finish");
        Assert.True(probe.Released, "the probe's dispose ran no cross-model release (no other live model?)");
        Assert.True(probe.HeldTheDecodingModelsLock, "the shared release ran without the retiring model's step lock");
        _output.WriteLine($"{backend}: a dispose beside a retiring model waited for the encode holding its lock");
    }

    /// <summary>A model with no weights whose dispose reports whether its cross-model release held another model's step
    /// lock.</summary>
    private sealed class ReleaseProbe : ModelBase
    {
        private readonly ModelBase _decoding;

        public ReleaseProbe(string path, BackendType backend, ModelBase decoding) : base(path, backend)
        {
            _decoding = decoding;
            Config = new ModelConfig { Architecture = "probe" };
        }

        public bool Released { get; private set; }
        public bool HeldTheDecodingModelsLock { get; private set; }

        internal override void InsideCrossModelReleaseForTests()
        {
            Released = true;
            HeldTheDecodingModelsLock = _decoding != null && Monitor.IsEntered(_decoding.GpuComputeLock);
        }

        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }

    /// <summary>The pairs whose kept model crashed the process on ggml_cuda (SIGSEGV in TSGgml_GraphRun) when the other was
    /// disposed beside it, and one that did not.</summary>
    [Theory]
    [InlineData("lfm3", "qwen35")]
    [InlineData("lfm3", "mistral3")]
    [InlineData("qwen35", "mistral3")]
    [InlineData("mistral3", "lfm3")]
    public void ALiveModelCarriesOnExactlyAfterAnotherIsDisposedBesideIt(string keptFamily, string otherFamily)
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        int[] tokens = { 41, 77, 102 };
        List<float[]> uninterrupted;
        using (ModelBase alone = ModelBase.Create(paths[keptFamily], backend))
            uninterrupted = DecodeSteps(alone, tokens, null);
        using ModelBase kept = ModelBase.Create(paths[keptFamily], backend);
        List<float[]> carried = DecodeSteps(kept, tokens, i =>
        {
            if (i == 1) LoadRunDispose(paths[otherFamily], backend);
        });
        AssertSame(uninterrupted, carried, $"{keptFamily} with {otherFamily} disposed beside it on {backend}");
    }
}
