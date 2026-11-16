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
/// An attention layer's query/key prep and cache writes (GraphFusion.AttentionPrep) on the CUDA graph executor:
/// ts_graph_attn_prep_f32 against the region's ops (StandardDecoderModel.EmitAttention's with Qwen35RuntimeModel.EmitQueryKey:
/// the queries' copy out of the gated projection, the weighted norms and rotaries of q and k, the copies that regroup a chunk's K
/// and V head-first, the K and V row writes), each run on its own copies of one pair of caches through the same steps at
/// positions a rotary delta past their KV indices - the rotated queries and the caches compared bit for bit after every step.
/// Decode rows and verify windows over Flash Next's layout (4 / 2 heads, separate K and V projections, F16 caches, NeoX rotary
/// over 64 of 256 dims), a packed q|gate|k|v projection read where it lies, an F32 cache under a YaRN rotary with interleaved
/// pairs over a head narrower than the norm's 256 threads, and a head wider than them; prompt chunks of 2048, 61 and 300 rows
/// across the 2048-row chunk boundary then decode rows and windows, over Qwen3.8-27B's layout (24 / 4 heads), with the queries
/// laid out as rows or head by head (the region's hint), and over a packed F32 / YaRN layout.
/// </summary>
public sealed class CudaAttentionPrepRegionTests
{
    private const int RopeDims = 64, RopeDelta = 7;
    private const float Eps = 1e-6f;

    private static IntPtr P(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    /// <summary>Decode rows and verify windows (first row, rows) in a 64-row cache.</summary>
    private static readonly (int Start, int Count)[] DecodeSteps = { (5, 1), (6, 1), (7, 3), (10, 8), (18, 1), (19, 2), (21, 5) };

    /// <summary>Prompt chunks across the default 2048-row chunk boundary, then decode rows and windows, in a 4096-row cache.</summary>
    private static readonly (int Start, int Count)[] PromptSteps = { (0, 2048), (2048, 61), (2109, 300), (2409, 1), (2410, 8), (2418, 61) };

    /// <summary>The rotary: dims, mode, original context, then base, frequency scale, extrapolation factor, attention factor,
    /// beta fast and beta slow (the region's order).</summary>
    private sealed record Rotary(long[] Ip, float[] Fp);

    private static readonly Rotary NeoX = new(new long[] { RopeDims, 2, 0 }, new[] { 1e7f, 1f, 0f, 1f, 0f, 0f });
    private static readonly Rotary Yarn = new(new long[] { RopeDims, 0, 4096 }, new[] { 10000f, 0.25f, 1f, 1f, 32f, 1f });

    /// <summary>A layer's shape: head width, query and KV heads, a packed q|gate|k|v projection, F16 caches, the rotary, and
    /// whether the region lets the queries go out head by head.</summary>
    private sealed record Layout(int D, int Heads, int KvHeads, bool Packed, bool F16, Rotary Rope, bool HeadMajor, int Capacity)
    {
        /// <summary>The projection's floats a row: the gated queries (each head's q then its gate), and when packed the keys and
        /// values after them.</summary>
        public int Width => 2 * Heads * D + (Packed ? 2 * KvHeads * D : 0);
    }

    [CudaTheory]
    [InlineData(256, 4, 2, false, true, false, false, false)]   // Qwen3.8 Flash Next's attention layers
    [InlineData(256, 4, 2, true, true, false, false, false)]    // a packed q|gate|k|v projection
    [InlineData(128, 4, 2, true, false, true, false, false)]    // F32 caches, YaRN, interleaved pairs; half the norm's threads idle
    [InlineData(512, 4, 2, false, true, false, false, false)]   // two values a thread in the norm
    [InlineData(256, 24, 4, false, true, false, true, true)]    // Qwen3.8-27B's prompt chunks, the queries head by head
    [InlineData(256, 24, 4, false, true, false, false, true)]   // the same as rows of heads
    [InlineData(128, 8, 2, true, false, true, true, true)]      // packed, F32 caches, YaRN, head by head
    public void PrepRegionWritesTheQueriesAndCacheBytesItsOpsWrite(int d, int heads, int kvHeads, bool packed, bool f16, bool yarn,
        bool headMajor, bool prompt)
    {
        var layout = new Layout(d, heads, kvHeads, packed, f16, yarn ? Yarn : NeoX, headMajor, prompt ? 4096 : 64);
        (int Start, int Count)[] steps = prompt ? PromptSteps : DecodeSteps;
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(d * 7 + heads + (packed ? 1 : 0) + (f16 ? 2 : 0) + (prompt ? 5 : 0));
            float[] F(int n, double lo, double hi)
            {
                var a = new float[n];
                for (int i = 0; i < n; i++) a[i] = (float)(lo + (hi - lo) * rng.NextDouble());
                return a;
            }
            // One starting cache pair for both runs: rows before the steps as the cache would hold them, the rest garbage.
            int cacheFloats = kvHeads * layout.Capacity * d;
            float[] kStart = F(cacheFloats, -2, 2).Select(x => f16 ? (float)(System.Half)x : x).ToArray();
            float[] vStart = F(cacheFloats, -2, 2).Select(x => f16 ? (float)(System.Half)x : x).ToArray();
            using var qNorm = Tensor.FromArray(allocator, F(d, 0.9, 1.1));
            using var kNorm = Tensor.FromArray(allocator, F(d, 0.9, 1.1));
            using Tensor opsK = Cache(allocator, kernels, kStart, layout), opsV = Cache(allocator, kernels, vStart, layout);
            using Tensor regionK = Cache(allocator, kernels, kStart, layout), regionV = Cache(allocator, kernels, vStart, layout);
            using var opsExecutor = new CudaGraphExecutor(allocator);
            using var regionExecutor = new CudaGraphExecutor(allocator);
            foreach ((int first, int count) in steps)
            {
                float[] projection = F(layout.Width * count, -3, 3);
                float[] keys = packed ? null : F(kvHeads * d * count, -3, 3);
                float[] values = packed ? null : F(kvHeads * d * count, -3, 3);
                float[] want = Run(opsExecutor, opsK, opsV, qNorm, kNorm, layout, projection, keys, values, first, count, region: false);
                float[] got = Run(regionExecutor, regionK, regionV, qNorm, kNorm, layout, projection, keys, values, first, count, region: true);
                string rows = $"rows {first}..{first + count - 1}";
                AssertSameBits(want, got, $"{rows}: query value", d);
                AssertSameBits(Floats(allocator, kernels, opsK), Floats(allocator, kernels, regionK), $"after {rows}: K cache value", d);
                AssertSameBits(Floats(allocator, kernels, opsV), Floats(allocator, kernels, regionV), $"after {rows}: V cache value", d);
            }
            Assert.Equal(0, opsExecutor.AttentionPrepRegions);
            Assert.Equal(steps.Length, regionExecutor.AttentionPrepRegions);
            Assert.Equal(headMajor ? steps.Count(s => s.Count > CudaDecodeDynParams.WindowRowBlocks) : 0, regionExecutor.HeadMajorQueryPreps);
        }
    }

    private static void AssertSameBits(float[] want, float[] got, string what, int d)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what} {i} (row {i / d}, column {i % d}) is {got[i]:R} from the region's kernel, {want[i]:R} from its ops");
    }

    /// <summary>A head-first cache [kvHeads, capacity, d] holding <paramref name="values"/> (F16 or F32).</summary>
    private static Tensor Cache(CudaAllocator allocator, CudaGraphKernels kernels, float[] values, Layout layout)
    {
        var f32 = Tensor.FromArray(allocator, values);
        if (!layout.F16)
        {
            using (f32) return f32.View(layout.KvHeads, layout.Capacity, layout.D);
        }
        using (f32)
        {
            var half = new Tensor(allocator, DType.Float16, layout.KvHeads, layout.Capacity, layout.D);
            foreach (Tensor t in new[] { f32, half }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            allocator.Context.MakeCurrent();
            kernels.LaunchConvert(P(f32), P(half), values.Length, true, allocator.Stream.Handle);
            allocator.Stream.Synchronize();
            ((CudaStorage)half.Storage).MarkDeviceModified();
            return half;
        }
    }

    private static float[] Floats(CudaAllocator allocator, CudaGraphKernels kernels, Tensor cache)
    {
        if (cache.ElementType == DType.Float32) return cache.GetElementsAsFloat((int)cache.ElementCount());
        using var f32 = new Tensor(allocator, DType.Float32, cache.ElementCount());
        foreach (Tensor t in new[] { cache, f32 }) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        allocator.Context.MakeCurrent();
        kernels.LaunchConvert(P(cache), P(f32), cache.ElementCount(), false, allocator.Stream.Handle);
        allocator.Stream.Synchronize();
        ((CudaStorage)f32.Storage).MarkDeviceModified();
        return f32.GetElementsAsFloat((int)f32.ElementCount());
    }

    /// <summary>One step's prep as the model emits it for one sequence (rows at their KV indices, rotated a delta past them), with
    /// or without the region around it: the rotated queries.</summary>
    private static float[] Run(CudaGraphExecutor executor, Tensor kCache, Tensor vCache, Tensor qNorm, Tensor kNorm, Layout layout,
        float[] projection, float[] keys, float[] values, int start, int count, bool region)
    {
        const long f = sizeof(float);
        int d = layout.D, heads = layout.Heads, kvHeads = layout.KvHeads, capacity = layout.Capacity;
        bool packed = layout.Packed, f16 = layout.F16;
        Rotary rope = layout.Rope;
        int width = layout.Width, cacheType = f16 ? GraphFormat.F16 : GraphFormat.F32;
        long qWidth = 2L * heads * d, kDim = (long)kvHeads * d;
        int[] positions = Enumerable.Range(start + RopeDelta, count).ToArray();
        long[] rows = Enumerable.Range(start, count).Select(r => (long)r).ToArray();

        var g = new GraphProgramBuilder(record: true);
        int projNode = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { width, count }, IntPtr.Zero, projection.Length * f);
        int keysNode = packed ? -1 : g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { kDim, count }, IntPtr.Zero, keys.Length * f);
        int valuesNode = packed ? -1 : g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { kDim, count }, IntPtr.Zero, values.Length * f);
        int qNormNode = g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { d }, IntPtr.Zero, d * f, handle: qNorm);
        int kNormNode = g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { d }, IntPtr.Zero, d * f, handle: kNorm);
        int kCacheNode = g.Leaf(GraphLeafRole.State, cacheType, new long[] { d, capacity, kvHeads }, GraphIdentity.Of(kCache.Storage),
            kCache.ElementCount() * (f16 ? 2 : 4), handle: kCache);
        int vCacheNode = g.Leaf(GraphLeafRole.State, cacheType, new long[] { d, capacity, kvHeads }, GraphIdentity.Of(vCache.Storage),
            vCache.ElementCount() * (f16 ? 2 : 4), handle: vCache);
        int posNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I32, new long[] { count }, IntPtr.Zero, count * 4L);
        int rowsNode = g.Leaf(GraphLeafRole.Input, GraphFormat.I64, new long[] { count, 1 }, IntPtr.Zero, count * 8L);

        // The projections' heads as EmitAttention views them (GraphStep.View3 records a view strided where its rows or heads are).
        (int q, bool qStrided) = View3(g, projNode, d, heads, count, 2L * d * f, width * f, 0);
        (int k, bool kStrided) = packed ? View3(g, projNode, d, kvHeads, count, d * f, width * f, qWidth * f)
            : (Reshape(g, keysNode, d, kvHeads, count), false);
        (int v, bool vStrided) = packed ? View3(g, projNode, d, kvHeads, count, d * f, width * f, (qWidth + kDim) * f)
            : (Reshape(g, valuesNode, d, kvHeads, count), false);
        if (region)
            g.BeginFused(GraphFusion.AttentionPrep, stackalloc[] { q, k, v, qNormNode, kNormNode, kCacheNode, vCacheNode, posNode, rowsNode },
                stackalloc long[] { heads, kvHeads, d, d, rope.Ip[0], rope.Ip[1], rope.Ip[2], layout.HeadMajor ? 1 : 0 },
                stackalloc[] { Eps, rope.Fp[0], rope.Fp[1], rope.Fp[2], rope.Fp[3], rope.Fp[4], rope.Fp[5] });
        // GraphStep.Cont copies a strided view; EmitQueryKey norms and rotates the copies.
        q = qStrided ? g.Op(GraphOp.Cont, stackalloc[] { q }) : q;
        k = kStrided ? g.Op(GraphOp.Cont, stackalloc[] { k }) : k;
        v = vStrided ? g.Op(GraphOp.Cont, stackalloc[] { v }) : v;
        q = Rope(g, Norm(g, q, qNormNode), posNode, rope);
        k = Rope(g, Norm(g, k, kNormNode), posNode, rope);
        int kFresh = count == 1 ? Reshape(g, k, d, 1, kvHeads) : g.Op(GraphOp.Cont, stackalloc[] { Permute(g, k) });
        int vFresh = count == 1 ? Reshape(g, v, d, 1, kvHeads) : g.Op(GraphOp.Cont, stackalloc[] { Permute(g, v) });
        g.Effect(g.Op(GraphOp.SetRows, stackalloc[] { kCacheNode, kFresh, rowsNode }));
        g.Effect(g.Op(GraphOp.SetRows, stackalloc[] { vCacheNode, vFresh, rowsNode }));
        if (region) g.EndFused(stackalloc[] { q });
        g.Output(q, (long)heads * d * count);

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
            var nodes = new List<int> { projNode, posNode, rowsNode };
            var data = new List<IntPtr> { Pin(projection), Pin(positions), Pin(rows) };
            var bytes = new List<long> { projection.Length * f, count * 4L, count * 8L };
            if (!packed)
            {
                nodes.AddRange(new[] { keysNode, valuesNode });
                data.AddRange(new[] { Pin(keys), Pin(values) });
                bytes.AddRange(new[] { keys.Length * f, values.Length * f });
            }
            Assert.True(plan.Run(nodes.ToArray(), data.ToArray(), bytes.ToArray()), plan.LastError);
            return (float[])plan.Outputs[0].Clone();
        }
        finally
        {
            foreach (GCHandle h in pins) h.Free();
        }
    }

    private static int Norm(GraphProgramBuilder g, int x, int weight)
        => g.Op(GraphOp.Mul, stackalloc[] { g.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { Eps }), weight });

    private static int Rope(GraphProgramBuilder g, int x, int positions, Rotary rope)
        => g.Op(GraphOp.RopeExt, stackalloc[] { x, positions }, rope.Ip, rope.Fp);

    private static int Reshape(GraphProgramBuilder g, int x, params long[] ne) => g.Op(GraphOp.Reshape, stackalloc[] { x }, ne);

    private static int Permute(GraphProgramBuilder g, int x) => g.Op(GraphOp.Permute, stackalloc[] { x }, stackalloc long[] { 0, 2, 1, 3 });

    private static (int Node, bool Strided) View3(GraphProgramBuilder g, int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
        => (g.Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset }),
            (ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4));
}
