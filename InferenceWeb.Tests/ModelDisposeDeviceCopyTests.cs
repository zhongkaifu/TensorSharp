// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// A GGML device copy is keyed by the address and size of its host bytes alone, and a disposed model's weight blocks can be
/// handed out again the moment they are freed. <see cref="ModelBase.Dispose"/> therefore clears the device-copy cache before
/// it frees its dense weights: bytes of the same size placed where a freed weight was (here the very block, from the model's
/// own pool, while its dispose is still running) bind a fresh copy of themselves, never the weight's. Only backends that
/// keep separate device copies (CUDA, Vulkan) can bind a stale one; elsewhere the weights are wrapped in place.
/// </summary>
public sealed class ModelDisposeDeviceCopyTests
{
    private const int In = 256, Out = 128, Rows = 4;   // a 128 KiB F32 weight

    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaBytesAtAFreedWeightsAddressNeverBindItsDeviceCopy() => Verify(BackendType.GgmlCuda, GgmlBackendType.Cuda);

    [GgmlFact(BackendType.GgmlVulkan)]
    public void VulkanBytesAtAFreedWeightsAddressNeverBindItsDeviceCopy() => Verify(BackendType.GgmlVulkan, GgmlBackendType.Vulkan);

    private static void Verify(BackendType backend, GgmlBackendType ggml)
    {
        string path = Path.Combine(Path.GetTempPath(), $"dispose-device-copy-{Guid.NewGuid():N}.gguf");
        SyntheticGguf.Write(path, [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }], []);
        try
        {
            var rng = new Random(11);
            float[] w = Enumerable.Range(0, In * Out).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] x = Enumerable.Range(0, In * Rows).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            float[] negated = w.Select(v => -v).ToArray();

            var model = new WeightModel(path, backend, w);
            IntPtr freed = model.WeightBytes;
            float[] first = Product(ggml, freed, x);
            Assert.Equal(In * Out * 4L, GgmlBasicOps.CachedBufferBytes(freed));   // the weight has its device copy

            IntPtr reused = IntPtr.Zero;
            float[] second = null;
            model.AfterWeightsFreed = () =>
            {
                // The negated values in a block of the same size: the pool hands back the one the weight just left.
                using var t = new Tensor(model.Allocator, DType.Float32, Out, In);
                reused = TensorComputePrimitives.GetStoragePointer(t);
                t.SetElementsAsFloat(negated);
                try { second = Product(ggml, reused, x); }
                finally { GgmlBasicOps.InvalidateHostBuffer(reused); }
            };
            model.Dispose();

            Assert.Equal(freed, reused);
            Assert.NotNull(second);
            // Bit for bit the negation (every rounding is symmetric in sign); the weight's own copy would give `first`.
            Assert.Equal(first.Select(v => -v), second);
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>[Rows, Out] = x · Wᵀ, the weight a graph leaf bound by its host bytes (as a model's programs bind theirs).</summary>
    private static float[] Product(GgmlBackendType backend, IntPtr weight, float[] x)
    {
        var g = new GraphProgramBuilder(record: true);
        int wn = g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { In, Out }, weight, In * Out * 4L);
        int xn = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { In, Rows }, IntPtr.Zero, x.Length * 4L);
        int y = g.Op(GraphOp.MulMat, wn, xn);
        g.Output(y, Out * Rows);
        using IGraphPlan plan = GgmlGraphExecutor.For(backend).Compile(g.ToProgram(replayable: true, reorder: false));
        var pin = System.Runtime.InteropServices.GCHandle.Alloc(x, System.Runtime.InteropServices.GCHandleType.Pinned);
        try
        {
            Assert.True(plan.Run(new[] { xn }, new[] { pin.AddrOfPinnedObject() }, new[] { x.Length * 4L }), plan.LastError);
            return (float[])plan.Outputs[0].Clone();
        }
        finally
        {
            pin.Free();
        }
    }

    /// <summary>A model with one dense weight on a GGML backend, which reports the point in its dispose where the weight is
    /// freed.</summary>
    private sealed class WeightModel : ModelBase
    {
        public WeightModel(string path, BackendType backend, float[] values) : base(path, backend)
        {
            Config = new ModelConfig { Architecture = "probe" };
            var w = new Tensor(_allocator, DType.Float32, Out, In);
            w.SetElementsAsFloat(values);
            _weights["probe.weight"] = w;
        }

        public IAllocator Allocator => _allocator;
        public IntPtr WeightBytes => TensorComputePrimitives.GetStoragePointer(_weights["probe.weight"]);
        public Action AfterWeightsFreed { get; set; }

        internal override void AfterDenseWeightsFreedForTests() => AfterWeightsFreed?.Invoke();
        protected override float[] ForwardCore(int[] tokens) => [];
        protected override void ResetKVCacheCore() { }
    }
}
