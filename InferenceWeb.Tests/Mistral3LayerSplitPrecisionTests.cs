using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class Mistral3LayerSplitPrecisionTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"mistral-layer-precision-{Guid.NewGuid():N}.gguf");
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);
        File.Delete(_path);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MixedPrecisionStagesChooseWholeCheckpointCachePrecision(bool explicitF32)
    {
        new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "mistral3", LayerCount = 4, HalfPrecisionLayer = 3,
        }.Write(_path);
        var expected = explicitF32 ? KvCacheDtype.F32 : KvCacheDtype.F16;
        // Reset the process default before each construction, as separate node
        // processes do. Parallel in-process construction would hide the defect
        // when the F16 stage changes the static default before the F32 stage.
        for (int node = 0; node < 2; node++)
        {
            KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, explicitF32);
            using var model = new InspectableMistral(_path, new MetadataGroup(node));
            Assert.Equal(expected, model.CacheDtype);
            // The first pipeline stage owns the leading F32 layers.
            Assert.Equal(LayerSplitPlan.StageOf(node, 2, LayerSplitPlan.DriverRunsLastStageFromEnvironment()) == 0,
                model.QuantizedCount == 0);
        }
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, explicitF32);
        using var reference = new InspectableMistral(_path);
        Assert.Equal(expected, reference.CacheDtype);
        Assert.All(reference.Forward([65, 66, 67]), value => Assert.True(float.IsFinite(value)));
        Assert.All(reference.Forward([68]), value => Assert.True(float.IsFinite(value)));
    }

    [Theory]
    [InlineData(KvCacheDtype.F32, false)]
    [InlineData(KvCacheDtype.F16, true)]
    [InlineData(KvCacheDtype.F32, true)]
    public void TensorParallelUsesTheSamePrecisionAsOneDevice(KvCacheDtype dtype, bool explicitlySet)
    {
        new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "mistral3", LayerCount = 4, HalfPrecisionLayer = 3,
        }.Write(_path);
        KvCacheDtypeConfig.RestoreForTests(dtype, explicitlySet);
        KvCacheDtype single;
        using (var reference = new InspectableMistral(_path))
            single = reference.CacheDtype;
        KvCacheDtypeConfig.RestoreForTests(dtype, explicitlySet);
        using var model = new InspectableMistral(_path, new CpuTensorGroup());

        // The KV cache precision is a property of the model, not of its placement: the
        // shared runtime resolves the default the same way for one device and for TP
        // (the old per-family TP path kept the raw captured default instead).
        Assert.Equal(single, model.CacheDtype);
        if (explicitlySet) Assert.Equal(dtype, model.CacheDtype);
        Assert.Equal(0, model.QuantizedCount);
        Assert.True(model.TensorParallelQuantizedCount > 0);
        Assert.All(model.KeyCachesForTests, layer =>
        {
            Assert.Equal(2, layer.Length);
            Assert.All(layer, cache =>
            {
                Assert.Equal(model.CacheDtype.ToDType(), cache.ElementType);
                Assert.Equal(DenseDecoderSyntheticModelBuilder.NumKvHeads / 2, cache.Sizes[0]);
            });
        });
    }

    private sealed class InspectableMistral(string path, ITensorParallelGroup group = null)
        : Mistral3Model(path, BackendType.Cpu, tpGroup: group)
    {
        public KvCacheDtype CacheDtype => _kvCacheDtype;
        public int QuantizedCount => _quantWeights.Count;
        public int TensorParallelQuantizedCount => _tpQuantWeights.Count;
    }

    // Only allocation and construction run here; no fake collective result can
    // accidentally qualify tensor-parallel inference.
    private sealed class CpuTensorGroup : ITensorParallelGroup
    {
        private readonly CpuAllocator[] _allocators = [new(BlasEnum.DotNet), new(BlasEnum.DotNet)];
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => _allocators[rank];
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void Dispose() { }
    }

    // Startup-only stand-in lets each stage construct independently, without
    // sharing the process-global dtype choice through a concurrent peer.
    private sealed class MetadataGroup(int node) : ILayerSplitGroup, INestedTensorParallelGroup
    {
        private readonly CpuAllocator _allocator = new(BlasEnum.DotNet);
        public int NodeId => node;
        public int NodeCount => 2;
        public int Degree => 1;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => node;
        public bool IsActive => false;
        public ITensorParallelGroup LocalGroup => this;
        public IAllocator GetAllocator(int rank) => _allocator;
        public void CrossNodeAllReduce(float[] buffer, int count)
        { for (int i = 0; i < count; i++) buffer[i] *= NodeCount; }
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void Dispose() { }
    }
}
