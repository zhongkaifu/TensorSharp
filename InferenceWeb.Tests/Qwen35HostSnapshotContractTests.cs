// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class Qwen35HostSnapshotContractTests
{
    [Fact]
    public void InvalidExtractionRanges_DoNotReadPointersOrModifyDestination()
    {
        using var fixture = new SnapshotFixture();
        byte[] destination = new byte[fixture.Model.ComputeKVBlockByteSize(4)];
        Array.Fill(destination, (byte)0xcd);
        foreach (var (start, count) in new[] { (-1, 4), (int.MaxValue, 4), (4, int.MaxValue), (5, 4), (0, 0), (0, -1) })
        {
            Assert.False(fixture.Model.TryExtractKVBlock(start, count, destination));
            Assert.All(destination, value => Assert.Equal((byte)0xcd, value));
        }
    }

    [Fact]
    public void InvalidLastRecurrentLayer_IsRefusedBeforeEarlierStateChanges()
    {
        using var fixture = new SnapshotFixture();
        byte[] original = fixture.Extract(0, 8);
        byte[] valid = fixture.Extract(0, 4);
        const int attentionBytes = 2 * 2 * 4 * 4 * sizeof(float); // K/V, heads, tokens, head dim.
        const int convBytes = 3 * 8 * sizeof(float);
        const int recurrentBytes = convBytes + sizeof(int) + 2 * 2 * 2 * sizeof(float);
        int finalWriteIndex = 2 * attentionBytes + recurrentBytes + convBytes;
        foreach (int invalid in new[] { -1, 3, int.MaxValue })
        {
            byte[] malformed = (byte[])valid.Clone();
            // Make earlier layers observably different if injection reaches them.
            Array.Fill(malformed, (byte)0x31, 0, attentionBytes);
            BinaryPrimitives.WriteInt32LittleEndian(malformed.AsSpan(finalWriteIndex), invalid);
            Assert.False(fixture.Model.TryInjectKVBlock(8, 4, malformed));
            Assert.Equal(original, fixture.Extract(0, 8));
        }
        foreach (int delta in new[] { -13, int.MaxValue })
        {
            byte[] malformed = (byte[])valid.Clone();
            BinaryPrimitives.WriteInt32LittleEndian(malformed.AsSpan(malformed.Length - sizeof(int)), delta);
            Assert.False(fixture.Model.TryInjectKVBlock(8, 4, malformed));
            Assert.Equal(original, fixture.Extract(0, 8));
        }
        Assert.True(fixture.Model.TryInjectKVBlock(8, 4, valid));
        Assert.Equal(valid, fixture.Extract(8, 4));
    }

    [Fact]
    public void IncompatibleRecurrentLayout_IsRefusedBeforeEarlierStateChanges()
    {
        using var fixture = new SnapshotFixture();
        fixture.Conv[3] = new float[23]; // Native/MLX import expects 3 * 8 conv values.
        byte[] original = fixture.Extract(0, 8);
        byte[] malformed = fixture.Extract(0, 4);
        malformed[0] ^= 0xff;
        Assert.False(fixture.Model.TryInjectKVBlock(8, 4, malformed));
        Assert.Equal(original, fixture.Extract(0, 8));
    }

    [Fact]
    public void ReuseCapability_IsLimitedToDenseCudaTrunkWithoutMtp()
    {
        using var fixture = new SnapshotFixture();
        Assert.False(fixture.Model.SupportsCrossSequenceKvReuse);
        fixture.SetBase("_backend", BackendType.GgmlCuda);
        Assert.True(fixture.Model.SupportsCrossSequenceKvReuse);
        fixture.Set("_numExperts", 1);
        Assert.False(fixture.Model.SupportsCrossSequenceKvReuse);
        fixture.Set("_numExperts", 0);
        fixture.Set("_numNextnLayers", 1);
        Assert.False(fixture.Model.SupportsCrossSequenceKvReuse);
        fixture.Set("_numNextnLayers", 0);
        fixture.SetBase("_backend", BackendType.GgmlMetal);
        Assert.False(fixture.Model.SupportsCrossSequenceKvReuse);
    }

    // Real Qwen snapshot methods over CPU-owned tensors, without loading weights
    // or invoking native kernels. GPU inference coverage belongs to ModelProbe.
    private sealed class SnapshotFixture : IDisposable
    {
        internal readonly Qwen35Model Model = (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        internal readonly float[][] Conv = new float[4][];
        private readonly List<Tensor> _owned = new();
        internal SnapshotFixture()
        {
            SetBase("<Config>k__BackingField", new ModelConfig
            { Architecture = "qwen35", NumLayers = 4, HiddenSize = 8, NumHeads = 2, NumKVHeads = 2 });
            SetBase("<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            SetBase("_backend", BackendType.Cpu);
            SetBase("_maxContextLength", 16);
            SetBase("_cacheSeqLen", 8);
            Set("_kvCacheCapacity", 16);
            Set("_isRecurrent", new[] { false, true, false, true });
            Set("_headKDim", 2); Set("_headVDim", 2);
            Set("_numKHeads", 1); Set("_numVHeads", 2); Set("_convKernel", 4);
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            var k = new Tensor[4]; var v = new Tensor[4]; var delta = new Tensor[4];
            for (int layer = 0; layer < 4; layer++)
            {
                if (layer % 2 == 0)
                {
                    k[layer] = Make(allocator, layer + 1, 2, 16, 4);
                    v[layer] = Make(allocator, layer + 5, 2, 16, 4);
                }
                else
                {
                    Conv[layer] = Enumerable.Range(0, 24).Select(i => layer + i * .125f).ToArray();
                    delta[layer] = Make(allocator, layer + 9, 2, 2, 2);
                }
            }
            Set("_kvCacheK", k); Set("_kvCacheV", v);
            Set("_convState", Conv); Set("_convStateWriteIdx", new[] { 0, 1, 0, 2 });
            Set("_deltaStateTensor", delta);
        }
        private Tensor Make(IAllocator allocator, int seed, params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            _owned.Add(tensor);
            tensor.SetElementsAsFloat(Enumerable.Range(0, (int)tensor.ElementCount()).Select(i => seed + i * .0625f).ToArray());
            return tensor;
        }
        internal byte[] Extract(int start, int count)
        {
            byte[] bytes = new byte[Model.ComputeKVBlockByteSize(count)];
            Assert.True(Model.TryExtractKVBlock(start, count, bytes));
            return bytes;
        }
        internal void Set(string name, object value) => SetField(typeof(Qwen35Model), name, value);
        internal void SetBase(string name, object value) => SetField(typeof(ModelBase), name, value);
        private void SetField(Type type, string name, object value) => type.GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(Model, value);
        public void Dispose()
        {
            foreach (var tensor in _owned) tensor.Dispose();
            GC.SuppressFinalize(Model);
        }
    }
}
