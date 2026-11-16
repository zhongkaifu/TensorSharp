// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A verify window's commit over a program's gated delta-net layers (GraphFusion.DeltaNetCommit) on the CUDA graph executor:
/// ts_graph_gdn_commit_f32 against the region's ops (Qwen35RuntimeModel.BuildCommitProgram's: per layer the re-step of the kept
/// rows - a GatedDeltaNet op over the saved inputs from the saved state, aliased into the backing, on the one-row kernel for
/// one row and the rows4 kernel for more - or the state copy, and the conv window copy), each run on its own copy of the same
/// layers' states. A window that updated its states in place is cut at every row (none kept: the saved state back; all kept:
/// only the conv windows); one wider than the in-place rows copies the state its verify buffer kept. The backings, states and
/// conv windows must be the same bytes, and the region must have served the commit.
/// </summary>
public sealed class CudaDeltaNetCommitRegionTests
{
    private const int ConvRows = 3, BackingRows = 8;

    /// <summary>One layer's recurrent state tensors (DeviceGdnState's): the conv window, the backing [rows' outputs | state],
    /// the saved state and scan inputs of an in-place window, the verify buffer of a wider one and the conv history.</summary>
    private sealed class Layer : IDisposable
    {
        public Tensor Conv, Backing, Saved, SavedInputs, Verify, ConvHistory;
        public int Capacity;

        public void Dispose()
        {
            foreach (Tensor t in new[] { Conv, Backing, Saved, SavedInputs, Verify, ConvHistory }) t?.Dispose();
        }
    }

    /// <summary>The launch's argument lays out as the kernel's struct ts_graph_gdn_commit (its static_assert holds 1,440 bytes):
    /// five arrays of 32 pointers, the 32 rows to step, then the eight shared shapes.</summary>
    [Fact]
    public void CommitArgumentMirrorsTheKernelsStruct()
    {
        Assert.Equal(1440, System.Runtime.CompilerServices.Unsafe.SizeOf<CudaGraphKernels.GdnCommitArgs>());
        Assert.Equal(1280, (int)System.Runtime.InteropServices.Marshal.OffsetOf<CudaGraphKernels.GdnCommitArgs>(nameof(CudaGraphKernels.GdnCommitArgs.Steps)));
        Assert.Equal(1408, (int)System.Runtime.InteropServices.Marshal.OffsetOf<CudaGraphKernels.GdnCommitArgs>(nameof(CudaGraphKernels.GdnCommitArgs.Layers)));
        Assert.Equal(1436, (int)System.Runtime.InteropServices.Marshal.OffsetOf<CudaGraphKernels.GdnCommitArgs>(nameof(CudaGraphKernels.GdnCommitArgs.ConvPitch)));
    }

    [CudaTheory]
    [InlineData(4, 0)]
    [InlineData(4, 1)]   // the ops' re-step runs the one-row kernel
    [InlineData(4, 2)]
    [InlineData(4, 3)]
    [InlineData(4, 4)]   // all kept: the conv windows alone
    [InlineData(8, 5)]
    [InlineData(2, 1)]
    [InlineData(16, 1)]  // wider than the in-place rows: the verify buffer's state after the kept rows
    [InlineData(16, 9)]
    [InlineData(16, 16)]
    public void CommitRegionWritesTheBytesItsOpsWrite(int rows, int kept) => AssertRegionMatchesOps(rows, kept, layers: 3, s: 128, vHeads: 6, kHeads: 2);

    /// <summary>More layers than one launch takes (32): the region launches twice.</summary>
    [CudaFact]
    public void CommitRegionOverMoreLayersThanALaunchTakesWritesItsOpsBytes() => AssertRegionMatchesOps(rows: 4, kept: 2, layers: 34, s: 32, vHeads: 2, kHeads: 1);

    /// <summary>A region the executor declines (heads of 16, not a multiple of 32) runs its ops when it closes: the writes it
    /// held - the re-steps' aliases and the conv windows' copies - in order, the bytes of the same ops without the region.</summary>
    [CudaFact]
    public void DeclinedCommitRegionRunsTheWritesItHeld() => AssertRegionMatchesOps(rows: 4, kept: 2, layers: 2, s: 16, vHeads: 2, kHeads: 1, served: false);

    private static void AssertRegionMatchesOps(int rows, int kept, int layers, int s, int vHeads, int kHeads, bool served = true)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        int channels = (2 * kHeads + vHeads) * s;
        var rng = new Random(29 + rows * 31 + kept);
        // One starting point for both: every buffer random, the gates' decays and the betas in their ranges.
        var start = Enumerable.Range(0, layers).Select(_ => Values(rng, rows, s, vHeads, kHeads, channels)).ToArray();
        var opsLayers = start.Select(v => Make(allocator, v, rows, s, vHeads, channels)).ToArray();
        var regionLayers = start.Select(v => Make(allocator, v, rows, s, vHeads, channels)).ToArray();
        try
        {
            using var opsExecutor = new CudaGraphExecutor(allocator);
            using var regionExecutor = new CudaGraphExecutor(allocator);
            Run(opsExecutor, opsLayers, rows, kept, s, vHeads, kHeads, channels, region: false);
            Run(regionExecutor, regionLayers, rows, kept, s, vHeads, kHeads, channels, region: true);
            Assert.Equal(0, opsExecutor.DeltaNetCommitRegions);
            Assert.Equal(served ? 1 : 0, regionExecutor.DeltaNetCommitRegions);
            for (int l = 0; l < layers; l++)
                foreach ((Tensor want, Tensor got, string what) in new[]
                {
                    (opsLayers[l].Backing, regionLayers[l].Backing, "backing"), (opsLayers[l].Conv, regionLayers[l].Conv, "conv window"),
                })
                {
                    float[] a = want.GetElementsAsFloat((int)want.ElementCount()), b = got.GetElementsAsFloat((int)got.ElementCount());
                    for (int i = 0; i < a.Length; i++)
                        if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
                            Assert.Fail($"rows {rows} kept {kept}: layer {l} {what} value {i} is {b[i]:R} from the region's kernel, {a[i]:R} from its ops");
                }
        }
        finally
        {
            foreach (Layer layer in opsLayers.Concat(regionLayers)) layer.Dispose();
        }
    }

    private sealed record LayerValues(float[] Conv, float[] Backing, float[] Saved, float[] SavedInputs, float[] Verify, float[] History);

    private static LayerValues Values(Random rng, int rows, int s, int vHeads, int kHeads, int channels)
    {
        long stateFloats = (long)vHeads * s * s, rowFloats = (long)vHeads * s;
        int capacity = Capacity(rows);
        float[] F(long n, double scale) => Enumerable.Range(0, (int)n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * scale)).ToArray();
        // Saved inputs, planar regions of BackingRows tokens: q | k [t][kHeads][s], v [t][vHeads][s], gates | betas [t][vHeads].
        long qk = (long)BackingRows * kHeads * s, v = (long)BackingRows * vHeads * s, gates = (long)BackingRows * vHeads;
        float[] inputs = F(2 * qk + v + 2 * gates, 0.2);
        for (long i = 0; i < gates; i++)
        {
            inputs[2 * qk + v + i] = (float)(-0.05 - rng.NextDouble());            // log decay
            inputs[2 * qk + v + gates + i] = (float)(0.05 + 0.9 * rng.NextDouble()); // beta
        }
        return new LayerValues(F((long)channels * ConvRows, 1), F(BackingRows * rowFloats + stateFloats, 0.5), F(stateFloats, 0.5), inputs,
            F(capacity * (rowFloats + stateFloats), 0.5), F((long)channels * (ConvRows + capacity), 1));
    }

    /// <summary>DeviceGdnState.EnsureSpecBuffers' capacity: the smallest power of two from 4 that holds the window.</summary>
    private static int Capacity(int rows)
    {
        int capacity = 4;
        while (capacity < rows) capacity *= 2;
        return capacity;
    }

    private static Layer Make(CudaAllocator allocator, LayerValues v, int rows, int s, int vHeads, int channels)
    {
        Tensor T(float[] values, params long[] sizes)
        {
            var t = new Tensor(allocator, DType.Float32, sizes);
            t.SetElementsAsFloat(values);
            return t;
        }
        int capacity = Capacity(rows);
        return new Layer
        {
            Capacity = capacity,
            Conv = T(v.Conv, channels, ConvRows),
            Backing = T(v.Backing, v.Backing.Length),
            Saved = T(v.Saved, v.Saved.Length),
            SavedInputs = T(v.SavedInputs, v.SavedInputs.Length),
            Verify = T(v.Verify, v.Verify.Length),
            ConvHistory = T(v.History, channels, ConvRows + capacity),
        };
    }

    private static int Leaf(GraphProgramBuilder g, Tensor t, params long[] ne)
        => g.Leaf(GraphLeafRole.State, GraphFormat.F32, ne, GraphIdentity.Of(t.Storage, t.StorageOffset), t.Storage.ByteLength, handle: t);

    private static int View(GraphProgramBuilder g, int x, params long[] ip) => g.Op(GraphOp.View, stackalloc[] { x }, ip);

    /// <summary>The commit as Qwen35RuntimeModel builds it on an executor whose states update in place (the backing holds the
    /// state), with or without the region around the layers' ops.</summary>
    private static void Run(CudaGraphExecutor executor, Layer[] layers, int rows, int kept, int s, int vHeads, int kHeads, int channels, bool region)
    {
        const long f = sizeof(float);
        long stateFloats = (long)vHeads * s * s, rowFloats = (long)vHeads * s, stateOffset = BackingRows * rowFloats;
        bool inPlace = rows <= BackingRows;
        var g = new GraphProgramBuilder(record: true);
        int n = layers.Length;
        var leaves = new (int Source, int Target, int Inputs, int History, int Conv, long Steps, long From, long To)[n];
        for (int l = 0; l < n; l++)
        {
            Layer layer = layers[l];
            int history = Leaf(g, layer.ConvHistory, ConvRows + layer.Capacity, channels), conv = Leaf(g, layer.Conv, ConvRows, channels);
            if (inPlace && kept >= rows) leaves[l] = (history, history, history, history, conv, -1, 0, 0);
            else if (inPlace)
            {
                int backing = Leaf(g, layer.Backing, stateOffset + stateFloats), saved = Leaf(g, layer.Saved, stateFloats);
                leaves[l] = kept == 0
                    ? (saved, backing, saved, history, conv, 0, 0, stateOffset)
                    : (saved, backing, Leaf(g, layer.SavedInputs, layer.SavedInputs.ElementCount()), history, conv, kept, 0, (BackingRows - kept) * rowFloats);
            }
            else
            {
                int verify = Leaf(g, layer.Verify, layer.Verify.ElementCount());
                leaves[l] = (verify, Leaf(g, layer.Backing, stateOffset + stateFloats), verify, history, conv, 0,
                    rowFloats * rows + (rows - kept) * stateFloats, stateOffset);
            }
        }
        if (region)
        {
            var ip = new long[8 + 3 * n];
            new long[] { s, vHeads, kHeads, BackingRows, ConvRows, channels, ConvRows + rows, kept }.CopyTo(ip, 0);
            for (int l = 0; l < n; l++)
            {
                ip[8 + 3 * l] = leaves[l].Steps;
                ip[9 + 3 * l] = leaves[l].From;
                ip[10 + 3 * l] = leaves[l].To;
            }
            g.BeginFused(GraphFusion.DeltaNetCommit,
                leaves.SelectMany(c => new[] { c.Source, c.Target, c.Inputs, c.History, c.Conv }).ToArray(), ip);
        }
        int last = -1;
        for (int l = 0; l < n; l++)
        {
            var c = leaves[l];
            if (inPlace && c.Steps > 0)
            {
                int before = View(g, c.Source, 4, s, s, vHeads, 1, s * f, s * s * f, stateFloats * f, 0);
                long qkRegion = (long)BackingRows * kHeads * s * f, vRegion = (long)BackingRows * vHeads * s * f, gRegion = (long)BackingRows * vHeads * f;
                int Region(long width, long heads, long at) => View(g, c.Inputs, 4, width, heads, kept, 1, width * f, width * heads * f, width * heads * kept * f, at);
                int q = Region(s, kHeads, 0), k = Region(s, kHeads, qkRegion), v = Region(s, vHeads, 2 * qkRegion);
                int gate = Region(1, vHeads, 2 * qkRegion + vRegion), beta = Region(1, vHeads, 2 * qkRegion + vRegion + gRegion);
                int gdn = g.Op(GraphOp.GatedDeltaNet, stackalloc[] { q, k, v, gate, beta, before }, stackalloc long[] { 1, 0, 1 });
                long total = rowFloats * kept + stateFloats, at = (BackingRows - kept) * rowFloats * f;
                int fallback = g.Op(GraphOp.Cpy, View(g, gdn, 1, total, 1, 1, 1, total * f, 0, 0, 0), View(g, c.Target, 1, total, 1, 1, 1, total * f, 0, 0, at));
                g.Alias(gdn, c.Target, at, fallback);
            }
            else if (c.Steps == 0)
            {
                int source = inPlace ? c.Source : View(g, c.Source, 1, stateFloats, 1, 1, 1, stateFloats * f, 0, 0, c.From * f);
                g.Effect(g.Op(GraphOp.Cpy, source, View(g, c.Target, 1, stateFloats, 1, 1, 1, stateFloats * f, 0, 0, c.To * f)));
            }
            int window = View(g, c.History, 2, ConvRows, channels, 1, 1, (ConvRows + rows) * f, 0, 0, kept * f);
            last = g.Op(GraphOp.Cpy, window, c.Conv);
            g.Effect(last);
        }
        if (region) g.EndFused(stackalloc[] { last });
        using IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false));
        Assert.True(plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty), plan.LastError);
        if (!region && inPlace && kept > 0 && kept < rows && s % 32 == 0)
            Assert.True(executor.AliasesInPlace >= n, $"the ops' re-steps did not write the backings in place ({executor.AliasesInPlace})");
    }
}
