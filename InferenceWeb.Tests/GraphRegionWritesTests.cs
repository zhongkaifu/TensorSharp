// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The tensor executor's session holds the writes inside a region its executor serves - effects and aliases alike (a commit
/// region's re-steps write the backings through aliases): when the kernel ran, none of them runs and none of the region's
/// ops; when it declined, they run in order as the region closes, an alias placing its result or running its copy, and
/// the state ends as the same ops without the region leave it.
/// </summary>
public sealed class GraphRegionWritesTests
{
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    /// <summary>An executor that serves <see cref="GraphFusion.DeltaNetCommit"/> regions (or declines them, <see cref="Serve"/>
    /// false) with a kernel that does nothing.</summary>
    private sealed class RegionExecutor : TensorGraphExecutor
    {
        public RegionExecutor() : base(Cpu) { }

        public bool Serve { get; init; }

        public int Served { get; private set; }

        protected override bool Fuses(GraphFusion kind) => kind == GraphFusion.DeltaNetCommit;

        protected override bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results)
        {
            if (!Serve) return false;
            results[0] = inputs[^1].CopyRef();
            Served++;
            return true;
        }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x) => throw new NotSupportedException();
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ServedRegionsSkipTheirWritesAndDeclinedOnesRunThemInOrder(bool serve)
    {
        float[] source = Enumerable.Range(0, 8).Select(i => 1f + i).ToArray();
        float[] Run(bool region, out float[] written, out int served)
        {
            using var src = Tensor.FromArray(Cpu, source);
            using var backing = Tensor.FromArray(Cpu, new float[16]);
            using var dst = Tensor.FromArray(Cpu, new float[8]);
            var g = new GraphProgramBuilder(record: true);
            int Leaf(Tensor t) => g.Leaf(GraphLeafRole.State, GraphFormat.F32, new[] { t.ElementCount() }, GraphIdentity.Of(t.Storage), t.ElementCount() * 4, handle: t);
            int s = Leaf(src), b = Leaf(backing), d = Leaf(dst);
            if (region) g.BeginFused(GraphFusion.DeltaNetCommit, stackalloc[] { s, b, d });
            // An aliased result: 2 * src into backing[4 .. 12); then a later write reading the backing it placed: dst = backing[4 .. 12) + src.
            int doubled = g.Op(GraphOp.Scale, stackalloc[] { s }, default, stackalloc[] { 2f });
            int placed = g.Op(GraphOp.View, stackalloc[] { b }, stackalloc long[] { 1, 8, 1, 1, 1, 32, 0, 0, 16 });
            g.Alias(doubled, b, 16, g.Op(GraphOp.Cpy, doubled, placed));
            int write = g.Op(GraphOp.Cpy, g.Op(GraphOp.Add, g.Op(GraphOp.View, stackalloc[] { b }, stackalloc long[] { 1, 8, 1, 1, 1, 32, 0, 0, 16 }), s), d);
            g.Effect(write);
            if (region) g.EndFused(stackalloc[] { write });
            var executor = new RegionExecutor { Serve = serve };
            using (IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false)))
                Assert.True(plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty), plan.LastError);
            written = dst.GetElementsAsFloat(8);
            served = executor.Served;
            return backing.GetElementsAsFloat(16);
        }

        float[] opsBacking = Run(region: false, out float[] opsDst, out int opsServed);
        float[] regionBacking = Run(region: true, out float[] regionDst, out int regionServed);
        Assert.Equal(0, opsServed);
        Assert.Equal(new float[] { 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0 }, opsBacking);
        Assert.Equal(source.Select(x => 3 * x).ToArray(), opsDst);
        if (serve)
        {
            // The kernel made the region's writes (here: none), so neither the alias nor the effect ran.
            Assert.Equal(1, regionServed);
            Assert.All(regionBacking, x => Assert.Equal(0f, x));
            Assert.All(regionDst, x => Assert.Equal(0f, x));
        }
        else
        {
            Assert.Equal(0, regionServed);
            Assert.Equal(opsBacking, regionBacking);
            Assert.Equal(opsDst, regionDst);
        }
    }
}
