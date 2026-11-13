// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A multi-node whole-layer pipeline whose nodes differ: each node runs its own number of stages, each stage a
// tensor-parallel group of its own size (--layer-split P --tp T per node, with --tp-peers), or the nodes have the
// same shape but different memory. The nodes exchange their shapes at startup, derive one plan (layers in proportion
// to capacity), and the pipeline matches one device through prefill, decode, a truncate-and-refill and a reset.
//
// The nodes run in this process over loopback TCP, their devices simulated (SimulatedDeviceGroup: independent host
// allocators, exact host reduction): functional evidence for the plan, the handshake, per-node sharding and
// stage transfers. GPU kernels and cross-machine transport are validated on real nodes separately.
using System.Net;
using System.Net.Sockets;

using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.Models.Placement;

namespace InferenceWeb.Tests;

public sealed class HeterogeneousPipelineTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-hetero-pipeline-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public HeterogeneousPipelineTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>(family, node shapes as "PxT" per node, each node's stage capacity; 0 = unmeasured).</summary>
    public static TheoryData<string, string, string> Topologies()
    {
        var data = new TheoryData<string, string, string>();
        foreach (string family in new[] { "mistral3", "qwen35" })
        {
            data.Add(family, "1x2,1x1", "0,0");          // two GPUs on the driver, one on the worker
            data.Add(family, "1x1,1x2", "0,0");
            data.Add(family, "2x1,1x2,1x1", "0,0,0");    // three nodes: two single-GPU stages, a TP stage, one GPU
            data.Add(family, "1x2,2x2", "0,0");          // a worker with two local TP stages
            data.Add(family, "1x1,1x1", "48,24");        // same shape, different memory: capacity decides
        }
        return data;
    }

    [Theory]
    [MemberData(nameof(Topologies))]
    public async Task NodesOfDifferentShapesMatchOneDevice(string family, string shapes, string capacities)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var nodeShapes = shapes.Split(',').Select(s => s.Split('x').Select(int.Parse).ToArray()).ToArray();
        long[] capacity = capacities.Split(',').Select(long.Parse).ToArray();
        int nodes = nodeShapes.Length;
        IPEndPoint[] endpoints = Endpoints(nodes);
        var devices = new SimulatedDeviceGroup[nodes];
        var models = await Task.WhenAll(Enumerable.Range(0, nodes).Select(node => Task.Run(() =>
        {
            int stages = nodeShapes[node][0], tp = nodeShapes[node][1];
            devices[node] = new SimulatedDeviceGroup(stages * tp);
            var group = new DistributedLayerSplitGroup(devices[node], node, endpoints, tp, capacity[node]);
            try { return Create(family, path, group); }
            catch { group.Dispose(); throw; }
        }))).WaitAsync(TimeSpan.FromSeconds(60));
        var workers = models.Skip(1).Select(model => Task.Run(model.RunDistributedWorkerLoop)).ToArray();
        try
        {
            // Every node agreed on one topology and owns its share of the layers, on its local mesh.
            var plans = models.Select(m => Assert.IsAssignableFrom<StandardDecoderModel>(m).Mesh.Plan).ToArray();
            for (int node = 0; node < nodes; node++)
            {
                DeviceMesh mesh = ((StandardDecoderModel)models[node]).Mesh;
                Assert.Equal(nodeShapes[node][0], mesh.Stages);
                Assert.Equal(nodeShapes[node][1], mesh.TensorParallel);
                Assert.Equal(nodeShapes[node][0], plans[node].LocalDegree);
                Assert.Equal(nodeShapes.Sum(s => s[0]), plans[node].GlobalDegree);
                Assert.True(plans[node].EndLayer - plans[node].FirstLayer >= nodeShapes[node][0], $"node {node} owns too few layers");
            }
            var owned = plans.OrderBy(p => p.StageIndex).ToArray();
            Assert.Equal(0, owned[0].FirstLayer);
            for (int p = 1; p < owned.Length; p++) Assert.Equal(owned[p - 1].EndLayer, owned[p].FirstLayer);
            Assert.Equal(Layers, owned[^1].EndLayer);
            if (capacity.All(c => c > 0) && capacity[0] > capacity[1])
                Assert.True(plans[0].EndLayer - plans[0].FirstLayer > plans[1].EndLayer - plans[1].FirstLayer,
                    $"the larger node owns {plans[0].EndLayer - plans[0].FirstLayer} layers, the smaller {plans[1].EndLayer - plans[1].FirstLayer}");

            List<float[]> reference;
            using (ModelBase single = Create(family, path, null))
                reference = Script(single, truncates: family != "qwen35");
            ModelBase driver = models[0];
            driver.BeginDistributedDriver();
            List<float[]> actual = await Task.Run(() => Script(driver, truncates: family != "qwen35")).WaitAsync(TimeSpan.FromSeconds(120));
            // Tensor-parallel stages split each row-parallel sum into per-rank partials: only the summation order changes.
            for (int i = 0; i < reference.Count; i++)
            {
                double rel = RelL2(actual[i], reference[i]);
                Assert.True(rel <= 1e-4, $"{family} {shapes} ({capacities}) row {i}: relL2 {rel:E2}");
            }
            ((DistributedLayerSplitGroup)GroupOf(driver)).BroadcastControl(4, []);
            await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(20));
            // A node of one tensor-parallel stage reduces through its own device group (several stages reduce through
            // sub-groups of it); single-device nodes never reduce.
            for (int node = 0; node < nodes; node++)
                if (nodeShapes[node][1] == 1) Assert.Equal(0, devices[node].AllReduceCount);
                else if (nodeShapes[node][0] == 1) Assert.True(devices[node].AllReduceCount > 0, $"node {node} never reduced");
        }
        finally
        {
            foreach (var model in models) model.Dispose();
        }
    }

    [Fact]
    public void AFamilyWithItsOwnPipelineRefusesNodesOfDifferentShapes()
    {
        var group = new TopologyOnlyGroup(stageTensorParallel: 2);
        var error = Assert.Throws<NotSupportedException>(() => new OwnPipelineModel(group));
        Assert.Contains("same number of single-GPU stages", error.Message);
    }

    private const int Layers = 8;

    private string Fixture(string family) => family switch
    {
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = Layers }
            .Write(Path.Combine(_dir, "mistral3.gguf")),
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = Layers, UntiedOutput = true }
            .Write(Path.Combine(_dir, "qwen35.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static ModelBase Create(string family, string path, ITensorParallelGroup group) => family switch
    {
        "mistral3" => new Mistral3Model(path, BackendType.Cpu, 1, group, 1),
        "qwen35" => new Qwen35RuntimeModel(path, BackendType.Cpu, 1, group, 1),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static ITensorParallelGroup GroupOf(ModelBase model)
        => (ITensorParallelGroup)typeof(ModelBase).GetProperty("LayerSplitGroup",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(model)!;

    /// <summary>Prefill, decode, a truncate-and-refill (not for recurrent layers, which truncate only to zero) and a
    /// reset; the driver mirrors each to its workers.</summary>
    private static List<float[]> Script(ModelBase model, bool truncates)
    {
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        if (truncates) model.TruncateKVCache(10);
        rows.Add((float[])model.ForwardRefill(new[] { 91, 92, 93, 94 }).Clone());
        rows.Add((float[])model.Forward(new[] { 96 }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37 }).Clone());
        rows.Add((float[])model.Forward(new[] { 40 }).Clone());
        return rows;
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

    /// <summary>A model that runs its own pipeline (no <c>RunsMixedNodeShapes</c>).</summary>
    private sealed class OwnPipelineModel(ITensorParallelGroup group) : ModelBase(null, BackendType.Cpu, tpGroup: group, openGguf: false)
    {
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    /// <summary>One node of a pipeline whose stages are two-GPU groups; never connects.</summary>
    private sealed class TopologyOnlyGroup(int stageTensorParallel) : ILayerSplitGroup
    {
        private readonly SimulatedDeviceGroup _devices = new(stageTensorParallel);
        public int NodeId => 0;
        public int StageTensorParallel => stageTensorParallel;
        public LayerSplitTopology Topology => new(new[] { 1, 1 }, new[] { stageTensorParallel, 1 }, new[] { 2L, 1L });
        public int Degree => _devices.Degree;
        public bool IsActive => false;
        public int GlobalDegree => Degree + 1;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => _devices.GetAllocator(rank);
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
        public void Dispose() => _devices.Dispose();
    }
}
