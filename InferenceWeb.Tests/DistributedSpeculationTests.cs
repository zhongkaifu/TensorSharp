// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Net;
using System.Net.Sockets;

using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Speculative decoding across the nodes of a multi-node layer split (loopback TCP between in-process nodes, simulated
/// devices): the driver drafts (it runs the last stage), every node verifies its layers over the window and keeps the
/// accepted prefix of its recurrent states, and the stream is plain greedy decoding on one device.
/// </summary>
public sealed class DistributedSpeculationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-dist-spec-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public DistributedSpeculationTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>(fixture, drafter, node shapes as "PxT" per node).</summary>
    public static TheoryData<string, string, string> Cases() => new()
    {
        { "dense", "scripted", "1x1,1x1" },
        { "dense", "scripted", "1x2,1x1" },          // the driver's stage is two tensor-parallel devices
        { "dense", "scripted", "1x1,1x2" },          // a worker's stage is
        { "dense", "scripted", "1x1,1x1,1x1" },      // three nodes
        { "dense-nextn", "draft-head", "1x1,1x1" },  // the MTP layer drafts on the driver's last stage
        { "dense-nextn", "draft-head", "1x2,1x1" },
        { "dense-nextn", "draft-head", "1x1,1x2" },
        { "dense", "dflash", "1x1,1x1" },            // the drafter's target layers' rows come from both nodes
        { "dense", "dflash", "1x1,1x1,1x1" },        // and from three
        { "dense", "dflash", "1x1,1x2" },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task PipelineSpeculationMatchesPlainGreedyOnOneDevice(string variant, string drafter, string shapes)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        string draft = drafter == "dflash"
            ? Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "dflash.gguf"), hidden: Qwen35DenseSyntheticModelBuilder.Hidden)
            : null;
        await using var cluster = await Cluster.StartAsync(path, shapes, draft);
        string label = $"qwen35 {variant} {drafter} over {shapes}";
        // Tensor-parallel stages split each row-parallel sum into per-rank partials: only near-ties may flip.
        double tie = shapes.Contains("x2") ? 1e-3 : 0;
        Func<ModelBase> reference = () => new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        await Task.Run(() =>
        {
            if (drafter == "draft-head")
                SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(cluster.TakeDriver, label, tieTolerance: tie, reference: reference);
            else if (drafter == "dflash")
                SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(cluster.TakeDriver, label, tieTolerance: tie, reference: reference);
            else
                SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(cluster.TakeDriver, label, tieTolerance: tie, reference: reference);
        }).WaitAsync(TimeSpan.FromSeconds(180));
        await cluster.StopWorkersAsync();
    }

    private const int Layers = 8;

    private string Fixture(string variant) => variant switch
    {
        "dense" => new Qwen35DenseSyntheticModelBuilder { Layers = Layers, UntiedOutput = true }.Write(Path.Combine(_dir, "dense.gguf")),
        "dense-nextn" => new Qwen35DenseSyntheticModelBuilder { Layers = Layers, UntiedOutput = true, NextN = true }
            .Write(Path.Combine(_dir, "dense-nextn.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    /// <summary>The nodes of one pipeline, each its own model over a loopback group; the workers mirror the driver.</summary>
    private sealed class Cluster : IAsyncDisposable
    {
        private readonly ModelBase[] _models;
        private readonly Task[] _workers;
        private bool _driverTaken;

        private Cluster(ModelBase[] models)
        {
            _models = models;
            _workers = models.Skip(1).Select(m => Task.Run(m.RunDistributedWorkerLoop)).ToArray();
        }

        public static async Task<Cluster> StartAsync(string path, string shapes, string draftModelPath = null)
        {
            int[][] nodeShapes = shapes.Split(',').Select(s => s.Split('x').Select(int.Parse).ToArray()).ToArray();
            IPEndPoint[] endpoints = Endpoints(nodeShapes.Length);
            ModelBase[] models = await Task.WhenAll(Enumerable.Range(0, nodeShapes.Length).Select(node => Task.Run(() =>
            {
                int stages = nodeShapes[node][0], tp = nodeShapes[node][1];
                var group = new DistributedLayerSplitGroup(new SimulatedDeviceGroup(stages * tp), node, endpoints, tp, 0);
                try { return (ModelBase)new Qwen35RuntimeModel(path, BackendType.Cpu, 1, group, 1, draftModelPath); }
                catch { group.Dispose(); throw; }
            }))).WaitAsync(TimeSpan.FromSeconds(60));
            models[0].BeginDistributedDriver();
            return new Cluster(models);
        }

        /// <summary>The driver, once (the check that takes it disposes it, which stops the workers).</summary>
        public ModelBase TakeDriver()
        {
            Assert.False(_driverTaken, "the driver was already taken");
            _driverTaken = true;
            return _models[0];
        }

        public async Task StopWorkersAsync() => await Task.WhenAll(_workers).WaitAsync(TimeSpan.FromSeconds(30));

        public async ValueTask DisposeAsync()
        {
            if (!_driverTaken) _models[0].Dispose();   // broadcasts the shutdown
            try { await Task.WhenAll(_workers).WaitAsync(TimeSpan.FromSeconds(30)); }
            catch { /* the test reports its own failure */ }
            foreach (ModelBase worker in _models.Skip(1)) worker.Dispose();
        }
    }

    private static IPEndPoint[] Endpoints(int count)
    {
        var listeners = Enumerable.Range(0, count).Select(_ => new TcpListener(IPAddress.Loopback, 0)).ToArray();
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }
}
