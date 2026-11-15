// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A QSA layer's indexer keys (GraphFusion.QsaKeys) on the CUDA graph executor: ts_graph_qsa_store_pool_f32 against the
/// region's ops (Qwen4ExpRuntimeModel.EmitAttentionInput's: the packed [raw | 0] rows and their write; the members' gather,
/// sum, scale, weighted norm and rotary, and the [last raw | pooled] write), each run on its own copy of one indexer cache
/// through the same steps - a prompt chunk crossing several blocks (its first block begun before it), decode rows that
/// complete a block and ones that do not (whose ops pool the last complete block again), and short chunks that complete
/// one block or none - the caches compared byte for byte after every step.
/// </summary>
public sealed class CudaQsaKeysRegionTests
{
    private const int D = 128, Ratio = 4, Capacity = 64, RopeDims = 64, RopeDelta = 7;
    private const float Eps = 1e-6f, FreqBase = 10000f;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    /// <summary>The steps (first row, rows): a 20-row prompt chunk from row 5, then decode rows and short chunks.</summary>
    private static readonly (int Start, int Count)[] Steps =
    {
        (5, 20), (25, 1), (26, 1), (27, 1), (28, 1), (29, 3), (32, 6), (38, 1), (39, 1), (40, 2),
    };

    [CudaFact]
    public void KeysRegionWritesTheCacheBytesItsOpsWrite()
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(41);
            // One starting cache for both: rows before the prompt as the cache would hold them (F16 values), the rest garbage.
            float[] start = Enumerable.Range(0, Capacity * 2 * D).Select(_ => (float)(System.Half)(rng.NextDouble() * 4 - 2)).ToArray();
            float[] norm = Enumerable.Range(0, D).Select(_ => (float)(0.9 + 0.2 * rng.NextDouble())).ToArray();
            using var normTensor = Tensor.FromArray(allocator, norm);
            using var opsCache = HalfCache(allocator, kernels, start);
            using var regionCache = HalfCache(allocator, kernels, start);
            using var opsExecutor = new CudaGraphExecutor(allocator);
            using var regionExecutor = new CudaGraphExecutor(allocator);
            foreach ((int first, int count) in Steps)
            {
                float[] raw = Enumerable.Range(0, count * D).Select(_ => (float)(rng.NextDouble() * 6 - 3)).ToArray();
                // A negative raw key's pooled half is -0 (0 times it): kept apart from +0 in the comparison below.
                raw[0] = -raw[0] - 1f;
                Run(opsExecutor, opsCache, normTensor, raw, first, count, region: false);
                Run(regionExecutor, regionCache, normTensor, raw, first, count, region: true);
                float[] want = Floats(allocator, kernels, opsCache), got = Floats(allocator, kernels, regionCache);
                for (int i = 0; i < want.Length; i++)
                    if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                        Assert.Fail($"after rows {first}..{first + count - 1}: cache row {i / (2 * D)} value {i % (2 * D)} is {got[i]:R} from the "
                            + $"region's kernel, {want[i]:R} from its ops");
            }
            Assert.Equal(0, opsExecutor.QsaKeysRegions);
            Assert.Equal(Steps.Length, regionExecutor.QsaKeysRegions);
        }
    }

    private static Tensor HalfCache(CudaAllocator allocator, CudaGraphKernels kernels, float[] values)
    {
        using var f32 = Tensor.FromArray(allocator, values);
        var half = new Tensor(allocator, DType.Float16, Capacity, 2 * D);
        foreach (Tensor t in new[] { f32, half }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(f32), P(half), values.Length, true, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)half.Storage).MarkDeviceModified();
        return half;
    }

    private static float[] Floats(CudaAllocator allocator, CudaGraphKernels kernels, Tensor half)
    {
        using var f32 = new Tensor(allocator, DType.Float32, half.ElementCount());
        foreach (Tensor t in new[] { half, f32 }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(half), P(f32), half.ElementCount(), false, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)f32.Storage).MarkDeviceModified();
        return f32.GetElementsAsFloat((int)f32.ElementCount());
    }

    /// <summary>One step's keys as the model emits them (QsaPoolSpan's blocks, text positions plus a rotary delta), with or
    /// without the region around them.</summary>
    private static void Run(CudaGraphExecutor executor, Tensor cache, Tensor norm, float[] raw, int start, int count, bool region)
    {
        const long f = sizeof(float);
        int end = (start + count) / Ratio;
        int firstBlock = Math.Max(count <= 16 ? end - (count + Ratio - 1) / Ratio : (start + Ratio) / Ratio - 1, 0), n = end - firstBlock;
        Assert.True(n > 0);
        int[] members = Enumerable.Range(0, Ratio * n).Select(i => (firstBlock + i / Ratio) * Ratio + i % Ratio).ToArray();
        int[] table = Enumerable.Range(0, n).Select(b => (firstBlock + b) * Ratio + RopeDelta).ToArray();
        long[] last = Enumerable.Range(0, n).Select(b => (long)((firstBlock + b) * Ratio + Ratio - 1)).ToArray();
        long[] rows = Enumerable.Range(start, count).Select(r => (long)r).ToArray();

        var g = new GraphProgramBuilder(record: true);
        int rawNode = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { D, count }, IntPtr.Zero, raw.Length * f);
        int cacheNode = g.Leaf(GraphLeafRole.State, GraphFormat.F16, new long[] { 2 * D, Capacity }, GraphIdentity.Of(cache.Storage),
            cache.ElementCount() * 2, handle: cache);
        int rowsNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I64, new long[] { count, 1 }, IntPtr.Zero, rows.Length * 8L);
        int membersNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I32, new long[] { members.Length }, IntPtr.Zero, members.Length * 4L);
        int tableNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I32, new long[] { n }, IntPtr.Zero, n * 4L);
        int lastNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I64, new long[] { n, 1 }, IntPtr.Zero, n * 8L);
        int normNode = g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { D }, IntPtr.Zero, D * f, handle: norm);
        if (region)
            g.BeginFused(GraphFusion.QsaKeys, stackalloc[] { rawNode, cacheNode, rowsNode, normNode, tableNode },
                stackalloc long[] { D, Ratio, count <= 16 ? 1 : 0, RopeDims, 2, 0 },
                stackalloc[] { Eps, 1f / Ratio, FreqBase, 1f, 0f, 1f, 0f, 0f });
        int zero = g.Op(GraphOp.Scale, stackalloc[] { rawNode }, default, stackalloc[] { 0f });
        int packed = g.Op(GraphOp.Concat, stackalloc[] { rawNode, zero }, stackalloc long[] { 0 });
        g.Effect(g.Op(GraphOp.SetRows, stackalloc[] { cacheNode, Reshape(g, packed, 2 * D, count, 1), rowsNode }));
        int rawHalf = View2(g, cacheNode, D, Capacity, 2L * D * 2, 0);
        int gathered = g.Op(GraphOp.GetRows, stackalloc[] { rawHalf, membersNode });
        int sum = -1;
        for (int i = 0; i < Ratio; i++)
        {
            int member = View2(g, gathered, D, n, Ratio * D * f, i * D * f);
            // GraphStep.Cont copies a view only where its rows are strided (more than one).
            sum = sum < 0 ? (n > 1 ? g.Op(GraphOp.Cont, stackalloc[] { member }) : member) : g.Op(GraphOp.Add, stackalloc[] { sum, member });
        }
        int pooled = g.Op(GraphOp.Scale, stackalloc[] { sum }, default, stackalloc[] { 1f / Ratio });
        pooled = g.Op(GraphOp.Mul, stackalloc[] { g.Op(GraphOp.RmsNorm, stackalloc[] { pooled }, default, stackalloc[] { Eps }), normNode });
        pooled = g.Op(GraphOp.RopeExt, stackalloc[] { Reshape(g, pooled, D, 1, n), tableNode }, stackalloc long[] { RopeDims, 2, 0 },
            stackalloc[] { FreqBase, 1f, 0f, 1f, 0f, 0f });
        int lastView = View2(g, gathered, D, n, Ratio * D * f, (Ratio - 1) * D * f);
        int lastRaw = n > 1 ? g.Op(GraphOp.Cont, stackalloc[] { lastView }) : lastView;
        int row = g.Op(GraphOp.Concat, stackalloc[] { lastRaw, Reshape(g, pooled, D, n) }, stackalloc long[] { 0 });
        int poolWrite = g.Op(GraphOp.SetRows, stackalloc[] { cacheNode, Reshape(g, row, 2 * D, n, 1), lastNode });
        g.Effect(poolWrite);
        if (region) g.EndFused(stackalloc[] { poolWrite });

        using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
        var pins = new List<GCHandle>();
        IntPtr Pin(Array a)
        {
            GCHandle h = GCHandle.Alloc(a, GCHandleType.Pinned);
            pins.Add(h);
            return h.AddrOfPinnedObject();
        }
        try
        {
            Assert.True(plan.Run(new[] { rawNode, rowsNode, membersNode, tableNode, lastNode },
                new[] { Pin(raw), Pin(rows), Pin(members), Pin(table), Pin(last) },
                new[] { raw.Length * f, rows.Length * 8L, members.Length * 4L, n * 4L, n * 8L }), plan.LastError);
        }
        finally
        {
            foreach (GCHandle h in pins) h.Free();
        }
    }

    private static int Reshape(GraphProgramBuilder g, int x, params long[] ne) => g.Op(GraphOp.Reshape, stackalloc[] { x }, ne);

    private static int View2(GraphProgramBuilder g, int x, long ne0, long ne1, long nb1, long offset)
        => g.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });
}
