using System.Net;
using System.Net.Sockets;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Distributed;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>CPU-only numerical tests of the shared ModelBase layer pipeline and
/// real multi-node control protocol. Synthetic stateful layers are not model-quality evidence.</summary>
public sealed class LayerSplitPipelineTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"layer-pipeline-{Guid.NewGuid():N}.gguf");
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    public LayerSplitPipelineTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        File.Delete(_path);
    }

    [Fact]
    public void UnevenPlansCoverEveryLayerExactlyOnceAndUseEveryDevice()
    {
        for (int nodes = 1; nodes <= 4; nodes++)
        for (int devices = 1; devices <= 4; devices++)
        for (int layers = nodes * devices; layers <= 64; layers++)
        {
            var owners = new int[layers];
            for (int node = 0; node < nodes; node++)
            {
                var plan = new LayerSplitPlan(layers, devices, node, nodes);
                var perDevice = new int[devices];
                for (int layer = plan.FirstLayer; layer < plan.EndLayer; layer++)
                {
                    Assert.True(plan.Owns(layer));
                    int rank = plan.LocalRank(layer);
                    Assert.InRange(rank, 0, devices - 1);
                    perDevice[rank]++;
                    owners[layer]++;
                }
                Assert.All(perDevice, count => Assert.True(count > 0));
                if (plan.FirstLayer > 0) Assert.Throws<ArgumentOutOfRangeException>(() => plan.LocalRank(plan.FirstLayer - 1));
                if (plan.EndLayer < layers) Assert.Throws<ArgumentOutOfRangeException>(() => plan.LocalRank(plan.EndLayer));
            }
            Assert.All(owners, count => Assert.Equal(1, count));
        }
        Assert.Throws<NotSupportedException>(() => new LayerSplitPlan(3, 2, 0, 2));
    }

    [Theory]
    [InlineData(2, 2, 7)]
    [InlineData(3, 2, 11)]
    public Task PipelineMatchesLocalPrefillDecodeRefillResetAndTruncate(int nodeCount, int localDegree, int layerCount)
        => RunPipelineParity(_path, nodeCount, localDegree, layerCount);

    /// <summary>Shared by the default (driver runs the last stage) and legacy
    /// (driver runs the first stage) orders; the plan is read like the model reads it.</summary>
    internal static async Task RunPipelineParity(string _path, int nodeCount, int localDegree, int layerCount)
    {
        bool driverLast = LayerSplitPlan.DriverRunsLastStageFromEnvironment();
        var endpoints = Endpoints(nodeCount);
        var models = await Task.WhenAll(Enumerable.Range(0, nodeCount).Select(rank => Task.Run(() =>
        {
            var group = new DistributedLayerSplitGroup(new CpuDevices(localDegree), rank, endpoints);
            try { return new StatefulModel(_path, layerCount, group); }
            catch { group.Dispose(); throw; }
        }))).WaitAsync(TimeSpan.FromSeconds(20));
        using var reference = new StatefulModel(_path, layerCount);
        var driver = models[0];
        var workers = models.Skip(1).Select(model => Task.Run(model.RunDistributedWorkerLoop)).ToArray();
        try
        {
            driver.BeginDistributedDriver();
            async Task Compare(int[] tokens, bool refill = false)
            {
                float[] expected = refill ? reference.ForwardRefill(tokens) : reference.Forward(tokens);
                float[] actual = await Task.Run(() => refill ? driver.ForwardRefill(tokens) : driver.Forward(tokens))
                    .WaitAsync(TimeSpan.FromSeconds(15));
                Assert.Equal(expected, actual); // lossless F32 boundaries permit exact parity on the same CPU
                Assert.Equal(9, actual.Length);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
            }

            await Compare([1, 5, 1017, 8]);
            await Compare([4]);
            await Compare([11]);
            reference.TruncateKVCache(3);
            driver.TruncateKVCache(3);
            await Compare([16, 18], refill: true);
            await Compare([3]);
            reference.ResetKVCache();
            driver.ResetKVCache();
            await Compare([21, 25]);
            await Compare([31, 4, 7], refill: true);
            await Compare([1]);

            driver.Group.BroadcastControl(4, []);
            await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(10));
            for (int rank = 0; rank < nodeCount; rank++)
            {
                var model = models[rank];
                Assert.Equal(reference.Controls, model.Controls);
                var plan = new LayerSplitPlan(layerCount, localDegree, rank, nodeCount, driverLast);
                Assert.Equal(plan.IsFirst ? 8 : 0, model.Embeddings);
                Assert.Equal(plan.IsLast ? 8 : 0, model.Projections);
                Assert.Equal(Enumerable.Range(plan.FirstLayer, plan.EndLayer - plan.FirstLayer)
                    .SelectMany(layer => Enumerable.Repeat(layer, 8)).Order(), model.ExecutedLayers.Order());
                Assert.All(model.ExecutedLayers, layer => Assert.True(plan.Owns(layer)));
                Assert.False(model.TensorParallelActive);
                for (int layer = 0; layer < layerCount; layer++)
                    Assert.Equal(plan.Owns(layer) ? 6 : 0, model.CachedTokens(layer));
            }
        }
        finally { foreach (var model in models) model.Dispose(); }
    }

    [Fact]
    public async Task DifferentModelMetadataIsRejectedBeforeForward()
    {
        var endpoints = Endpoints(2);
        var outcomes = await Task.WhenAll(Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
        {
            using var group = new DistributedLayerSplitGroup(new CpuDevices(2), rank, endpoints);
            return Record.Exception(() =>
            {
                using var model = new StatefulModel(_path, rank == 0 ? 7 : 8, group);
            });
        }))).WaitAsync(TimeSpan.FromSeconds(20));
        Assert.All(outcomes, error =>
        {
            Assert.IsType<InvalidOperationException>(error);
            Assert.Contains("different model metadata", error.Message);
        });
    }

    [Theory]
    [InlineData(false, 5, false)]
    [InlineData(true, 7, false)]
    [InlineData(false, 5, true)]
    public async Task RealMistralFixtureMatchesLocalAndLoadsOnlyOwnedWeights(bool tiedOutput, int layerCount, bool mixedPrecision)
    {
        if (mixedPrecision) KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, explicitlySet: false);
        new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "mistral3", LayerCount = layerCount, TiedOutput = tiedOutput,
            HalfPrecisionLayer = mixedPrecision ? layerCount - 1 : null,
        }.Write(_path);
        var endpoints = Endpoints(2);
        // CPU devices exercise the real architecture and weight-routing logic;
        // device-specific kernels are qualified separately on the remote GPUs.
        var models = await Task.WhenAll(Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
        {
            var group = new DistributedLayerSplitGroup(new CpuDevices(2), rank, endpoints);
            try { return new InspectableMistral(_path, group); }
            catch { group.Dispose(); throw; }
        }))).WaitAsync(TimeSpan.FromSeconds(30));
        // The same compute path as the pipeline (which runs per-op until multi-node stages are
        // programs), so the comparison stays exact: this test is about routing, not kernels.
        using var reference = new InspectableMistral(_path) { GraphLoweringEnabled = false };
        var driver = models[0];
        var worker = Task.Run(models[1].RunDistributedWorkerLoop);
        try
        {
            for (int node = 0; node < 2; node++)
            {
                var model = models[node];
                var plan = model.Placement;
                foreach (var (name, allocator) in model.LoadedWeights)
                {
                    if (LayerSplitPlan.TryGetLayer(name, out int layer))
                    {
                        Assert.True(plan.Owns(layer), $"node {node} loaded {name}");
                        Assert.Same(model.Group.GetAllocator(plan.LocalRank(layer)), allocator);
                    }
                    else if (name.StartsWith("output", StringComparison.Ordinal))
                    {
                        Assert.True(plan.IsLast);
                        Assert.Same(model.Group.GetAllocator(1), allocator);
                    }
                    else if (name == "token_embd.weight")
                    {
                        Assert.True(plan.IsFirst || (plan.IsLast && tiedOutput));
                        Assert.Same(model.Group.GetAllocator(plan.IsFirst ? 0 : 1), allocator);
                    }
                }
                foreach (var name in model.QuantizedWeightNames)
                    if (LayerSplitPlan.TryGetLayer(name, out int layer)) Assert.True(plan.Owns(layer));
                Assert.Equal(plan.EndLayer - plan.FirstLayer,
                    model.LoadedWeights.Keys.Concat(model.QuantizedWeightNames)
                        .Count(name => name.EndsWith(".attn_qkv.weight", StringComparison.Ordinal)));
                Assert.False(model.BatchedForwardAvailable);
                Assert.False(model.SupportsKVStateSnapshot);
            }

            driver.BeginDistributedDriver();
            async Task Compare(int[] tokens, bool refill = false)
            {
                var expected = refill ? reference.ForwardRefill(tokens) : reference.Forward(tokens);
                var actual = await Task.Run(() => refill ? driver.ForwardRefill(tokens) : driver.Forward(tokens))
                    .WaitAsync(TimeSpan.FromSeconds(20));
                Assert.Equal(expected, actual);
                Assert.All(actual, value => Assert.True(float.IsFinite(value)));
                Assert.Equal(reference.CacheSeqLen, driver.CacheSeqLen);
            }

            await Compare([65, 66, 67, 68, 69, 70, 71, 72, 73]);
            await Compare([74]);
            driver.TruncateKVCache(4);
            reference.TruncateKVCache(4);
            await Compare([75, 76], refill: true);
            await Compare([77]);
            driver.ResetKVCache();
            reference.ResetKVCache();
            await Compare([12, 21, 22], refill: true);
            await Compare([23]);
            driver.Group.BroadcastControl(4, []);
            await worker.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(reference.CacheSeqLen, models[1].CacheSeqLen);
        }
        finally { foreach (var model in models) model.Dispose(); }
    }

    [Fact]
    public async Task StageComputeFailureReleasesPeersWaitingForActivations()
    {
        var endpoints = Endpoints(2);
        var models = await Task.WhenAll(Enumerable.Range(0, 2).Select(rank => Task.Run(() =>
            new StatefulModel(_path, 7, new DistributedLayerSplitGroup(new CpuDevices(2), rank, endpoints)))))
            .WaitAsync(TimeSpan.FromSeconds(20));
        try
        {
            // Fail inside the worker's own stage, whichever order is configured.
            models[1].FailureLayer = models[1].Placement.FirstLayer;
            models[0].BeginDistributedDriver();
            var worker = Task.Run(() => Record.Exception(models[1].RunDistributedWorkerLoop));
            var driver = Task.Run(() => Record.Exception(() => models[0].Forward([7, 12])));
            var errors = await Task.WhenAll(driver, worker).WaitAsync(TimeSpan.FromSeconds(5));
            Assert.All(errors, error => Assert.NotNull(error));
            Assert.Contains("deliberate stage failure", errors[1].Message);
            Assert.All(models, model => Assert.False(((CpuDevices)model.Group.LocalGroup).IsDisposed));
        }
        finally { foreach (var model in models) model.Dispose(); }
    }

    private sealed class InspectableMistral(string path, DistributedLayerSplitGroup group = null)
        : Mistral3Model(path, BackendType.Cpu, tpGroup: group)
    {
        public DistributedLayerSplitGroup Group => group;
        public LayerSplitPlan Placement => LayerPlacement;
        public IReadOnlyDictionary<string, IAllocator> LoadedWeights => _weights.ToDictionary(pair => pair.Key, pair => pair.Value.Allocator);
        public IEnumerable<string> QuantizedWeightNames => _quantWeights.Keys;
    }

    private sealed class StatefulModel : ModelBase
    {
        private const int Width = 4;
        private readonly List<float[]>[] _states;
        public DistributedLayerSplitGroup Group { get; }
        public List<int> ExecutedLayers { get; } = [];
        public List<string> Controls { get; } = [];
        public int Embeddings { get; private set; }
        public int Projections { get; private set; }
        public LayerSplitPlan Placement => LayerPlacement;
        public bool TensorParallelActive => IsTensorParallel;
        public int FailureLayer { get; set; } = -1;
        public int CachedTokens(int layer) => _states[layer].Count;

        public StatefulModel(string path, int layerCount, DistributedLayerSplitGroup group = null)
            : base(path, BackendType.Cpu, tpGroup: group)
        {
            Group = group;
            Config = new ModelConfig { Architecture = "probe", NumLayers = layerCount, HiddenSize = Width, VocabSize = 9 };
            _states = Enumerable.Range(0, layerCount).Select(_ => new List<float[]>()).ToArray();
            try { ConfigureLayerPlacement("stateful-test-layers"); }
            catch { Dispose(); throw; }
        }

        protected override float[] ForwardCore(int[] tokens) => Run(tokens, "forward");
        protected override float[] ForwardRefillCore(int[] tokens) => Run(tokens, "refill");
        private float[] Run(int[] tokens, string operation)
        {
            Controls.Add(operation + ":" + string.Join(',', tokens));
            if (LayerPlacement != null)
                return ForwardLayerPipeline(tokens, () => Embed(tokens), RunLayer, Project);
            using var hidden = Embed(tokens);
            for (int layer = 0; layer < Config.NumLayers; layer++) RunLayer(hidden, layer);
            return Project(hidden);
        }

        private Tensor Embed(int[] tokens)
        {
            Embeddings++;
            var hidden = new Tensor(_allocator, DType.Float32, tokens.Length, Width);
            hidden.SetElementsAsFloat(tokens.SelectMany(token => Enumerable.Range(0, Width)
                .Select(feature => (token % 37) * 0.017f + feature * 0.03f)).ToArray());
            return hidden;
        }

        private Tensor RunLayer(Tensor hidden, int layer)
        {
            if (layer == FailureLayer) throw new InvalidOperationException("deliberate stage failure");
            ExecutedLayers.Add(layer);
            var values = hidden.GetElementsAsFloat((int)hidden.ElementCount());
            for (int token = 0; token < values.Length / Width; token++)
            {
                float[] prior = _states[layer].Count == 0 ? new float[Width] : _states[layer][^1];
                var next = new float[Width];
                for (int feature = 0; feature < Width; feature++)
                    next[feature] = MathF.Tanh(values[token * Width + feature] * (0.53f + layer * 0.07f)
                        + values[token * Width + (feature + 1) % Width] * 0.13f
                        + prior[feature] * 0.11f + (layer + 1) * 0.019f);
                _states[layer].Add(next);
                next.CopyTo(values, token * Width);
            }
            hidden.SetElementsAsFloat(values);
            return hidden;
        }

        private float[] Project(Tensor hidden)
        {
            Projections++;
            var values = hidden.GetElementsAsFloat((int)hidden.ElementCount());
            return Enumerable.Range(0, Config.VocabSize).Select(token =>
                values[values.Length - Width + token % Width] * (token + 1) - token * 0.023f).ToArray();
        }

        protected override void ResetKVCacheCore()
        {
            Controls.Add("reset");
            foreach (var state in _states) state.Clear();
        }
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            Controls.Add("truncate:" + tokenCount);
            foreach (var state in _states)
                if (state.Count > tokenCount) state.RemoveRange(tokenCount, state.Count - tokenCount);
        }
    }

    private sealed class CpuDevices(int degree) : ITensorParallelGroup
    {
        private readonly CpuAllocator[] _allocators = Enumerable.Range(0, degree).Select(_ => new CpuAllocator(BlasEnum.DotNet)).ToArray();
        public int Degree => degree;
        public bool IsActive => false;
        public bool IsDisposed { get; private set; }
        public int GlobalDegree => degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => _allocators[rank];
        public void AllReduce(Tensor[] tensors) => throw new InvalidOperationException("Layer execution must never use a local reduction.");
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => IsDisposed = true;
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
