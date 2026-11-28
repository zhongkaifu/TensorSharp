// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Eng.GraphSchema;
using TensorSharp.GGML;
using TensorSharp.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// The graph op protocol has one definition (TensorSharp.Core/Graph/Schema/graph-ops.json): the IR's GraphOp, the GGML bridge's
/// GgmlGraphOp and the native TsgGraphOp are generated from it, programs are checked against it as they are built, and the
/// native bridge refuses a managed side built from another schema.
/// </summary>
public sealed class GraphOpSchemaTests
{
    private static string RepoRoot()
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here != null && !File.Exists(Path.Combine(here.FullName, "TensorSharp.slnx")))
            here = here.Parent;
        return here?.FullName ?? throw new DirectoryNotFoundException("no repository root above " + AppContext.BaseDirectory);
    }

    private static GraphOpSchemaModel Schema()
        => GraphOpSchemaGenerator.Parse(File.ReadAllText(Path.Combine(RepoRoot(), GraphOpSchemaGenerator.SchemaPath)));

    [Fact]
    public void GeneratedDefinitionsAreCurrent()
    {
        GraphOpSchemaModel schema = Schema();
        foreach (KeyValuePair<string, string> file in GraphOpSchemaGenerator.Generate(schema))
        {
            string path = Path.Combine(RepoRoot(), file.Key);
            Assert.True(File.Exists(path), $"{file.Key} is missing: run `dotnet run --project eng/graph-schema`");
            string committed = File.ReadAllText(path).Replace("\r\n", "\n");
            Assert.True(committed == file.Value,
                $"{file.Key} is stale against {GraphOpSchemaGenerator.SchemaPath}: run `dotnet run --project eng/graph-schema`");
        }
    }

    [Fact]
    public void CompiledSchemaIsTheSchemaFile()
    {
        GraphOpSchemaModel schema = Schema();
        Assert.Equal(schema.Hash, GraphOpSchema.Hash);
        Assert.Equal(schema.Version, GraphOpSchema.Version);
        Assert.Equal(schema.Ops.Select(o => o.Code).OrderBy(c => c), Enum.GetValues<GraphOp>().Select(o => (int)o).OrderBy(c => c));
        foreach (GraphOpDefinition op in schema.Ops)
        {
            GraphOpSignature s = GraphOpSchema.Signature((GraphOp)op.Code);
            Assert.Equal(op.Name, s.Op.ToString());
            Assert.Equal((op.Srcs.Min, op.Srcs.Max, op.Ip.Min, op.Ip.Max, op.Fp.Min, op.Fp.Max),
                (s.SrcsMin, s.SrcsMax, s.IpMin, s.IpMax, s.FpMin, s.FpMax));
            Assert.True(Enum.TryParse(op.Name, out GgmlGraphOp mirrored), $"GgmlGraphOp has no {op.Name}");
            Assert.Equal(op.Code, (int)mirrored);
        }
        Assert.Equal(GraphOpSchema.Ops.Length, Enum.GetValues<GgmlGraphOp>().Length);
    }

    [Fact]
    public void HashCoversStructureNotDocs()
    {
        string json = File.ReadAllText(Path.Combine(RepoRoot(), GraphOpSchemaGenerator.SchemaPath));
        string hash = GraphOpSchemaGenerator.Parse(json).Hash;
        Assert.Equal(hash, GraphOpSchemaGenerator.Parse(json.Replace("\"doc\": \"ggml_exp.\"", "\"doc\": \"e to the x.\"")).Hash);
        Assert.NotEqual(hash, GraphOpSchemaGenerator.Parse(json.Replace("\"name\": \"Exp\", \"code\": 19", "\"name\": \"Exp\", \"code\": 76")).Hash);
        Assert.NotEqual(hash, GraphOpSchemaGenerator.Parse(json.Replace("\"slots\": [\"S=1\"]", "\"slots\": [\"S=2\"]")).Hash);
    }

    [Fact]
    public void SchemaRefusesInconsistentDefinitions()
    {
        string json = File.ReadAllText(Path.Combine(RepoRoot(), GraphOpSchemaGenerator.SchemaPath));
        Assert.Throws<FormatException>(() => GraphOpSchemaGenerator.Parse(json.Replace("\"code\": 20", "\"code\": 19")));
        Assert.Throws<FormatException>(() => GraphOpSchemaGenerator.Parse(json.Replace("[\"x\", \"positions\", \"freqFactors?\"]", "[\"x\", \"positions?\", \"freqFactors\"]")));
    }

    [Theory]
    [InlineData(GraphOp.Add, 1, 0, 0, "operands")]
    [InlineData(GraphOp.Add, 3, 0, 0, "operands")]
    [InlineData(GraphOp.RopeExt, 2, 2, 6, "integer parameters")]
    [InlineData(GraphOp.RopeExt, 2, 3, 5, "float parameters")]
    [InlineData(GraphOp.View, 1, 8, 0, "integer parameters")]
    [InlineData(GraphOp.KvWindow, 1, 5, 0, "integer parameters")]
    [InlineData(GraphOp.FlashAttnExt, 6, 0, 0, "operands")]
    [InlineData(GraphOp.Scale, 1, 0, 2, "float parameters")]
    public void BuilderRefusesAMalformedOpAsItIsRecorded(GraphOp op, int srcs, int ip, int fp, string what)
    {
        foreach (bool record in new[] { true, false })
        {
            var b = new GraphProgramBuilder(record);
            int x = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 4 }, IntPtr.Zero, 16);
            int[] s = Enumerable.Repeat(x, srcs).ToArray();
            var ex = Assert.Throws<ArgumentException>(() => b.Op(op, s, new long[ip], new float[fp]));
            Assert.Contains(op.ToString(), ex.Message);
            Assert.Contains(what, ex.Message);
            Assert.Equal(1, b.NodeCount);   // nothing was recorded
        }
    }

    [Fact]
    public void BuilderRefusesAnOpOutsideTheSchema()
    {
        var b = new GraphProgramBuilder(true);
        int x = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 4 }, IntPtr.Zero, 16);
        Assert.Throws<ArgumentException>(() => b.Op((GraphOp)60, stackalloc[] { x }));
        Assert.False(GraphOpSchema.IsDefined((GraphOp)60));
        Assert.False(GraphOpSchema.IsDefined(0));
    }

    [Fact]
    public void SlotConstantsNameTheSchemasPositions()
    {
        Assert.Equal(0, GraphOpParams.RopeExt.Ip.NDims);
        Assert.Equal(2, GraphOpParams.RopeExt.Ip.NCtxOrig);
        Assert.Equal(5, GraphOpParams.RopeExt.Fp.BetaSlow);
        Assert.Equal(10000f, GraphOpParams.RopeExt.Fp.DefaultFreqBase);
        Assert.Equal(2, GraphOpParams.RopeExt.Srcs.FreqFactors);
        Assert.Equal(3, GraphOpParams.FlashAttnExt.Ip.Band);
        Assert.Equal(4, GraphOpParams.FlashAttnExt.Srcs.Sinks);
        Assert.Equal(8, GraphOpParams.View.Ip.Offset);
        Assert.Equal(7, GraphOpParams.KvWindow.Ip.GqaRatio);
        Assert.Equal(1L, GraphOpParams.KvWindow.Ip.DefaultQueryRows);
        Assert.Equal(float.NegativeInfinity, GraphOpParams.Clamp.Fp.DefaultMin);
        Assert.Equal((6, 11), (GraphOpParams.MoeHostExperts.Srcs.MinCount, GraphOpParams.MoeHostExperts.Srcs.MaxCount));
    }

    /// <summary>The typed emitters record exactly what an emitter wrote by hand: same nodes, parameters and signature.</summary>
    [Fact]
    public void TypedEmittersMatchHandWrittenOps()
    {
        static (GraphProgramBuilder, int, int, int, int) Start()
        {
            var b = new GraphProgramBuilder(true);
            int q = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 8, 4, 2 }, IntPtr.Zero, 256);
            int k = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 8, 4, 2 }, IntPtr.Zero, 256);
            int v = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 8, 4, 2 }, IntPtr.Zero, 256);
            int m = b.Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { 4, 4 }, IntPtr.Zero, 32);
            return (b, q, k, v, m);
        }

        void Same(Action<GraphProgramBuilder, int, int, int, int> byHand, Action<GraphProgramBuilder, int, int, int, int> typed)
        {
            var (a, q1, k1, v1, m1) = Start();
            var (t, q2, k2, v2, m2) = Start();
            byHand(a, q1, k1, v1, m1);
            typed(t, q2, k2, v2, m2);
            Assert.Equal(a.Signature, t.Signature);
            GraphNode x = a.ToProgram(false, false).Entries[^1], y = t.ToProgram(false, false).Entries[^1];
            Assert.Equal(x.Op, y.Op);
            Assert.Equal(x.Srcs, y.Srcs);
            Assert.Equal(x.Ip, y.Ip);
            Assert.Equal(x.Fp, y.Fp);
        }

        // GraphEmitter.FlashAttention's four forms.
        Same((b, q, k, v, m) => b.Op(GraphOp.FlashAttnExt, stackalloc[] { q, k, v, m }, stackalloc long[] { 10 }, stackalloc[] { 0.5f, 0f, 0f }),
            (b, q, k, v, m) => b.EmitFlashAttnExt(q, k, v, m, accPrecision: 10, scale: 0.5f, softcap: 0f));
        Same((b, q, k, v, m) => b.Op(GraphOp.FlashAttnExt, stackalloc[] { q, k, v, m }, stackalloc long[] { 10, 3 }, stackalloc[] { 0.5f, 0f, 0f }),
            (b, q, k, v, m) => b.EmitFlashAttnExt(q, k, v, m, accPrecision: 10, kvMax: 3, scale: 0.5f, softcap: 0f));
        Same((b, q, k, v, m) => b.Op(GraphOp.FlashAttnExt, stackalloc[] { q, k, v, m }, stackalloc long[] { 10, 0, GraphAttentionRows.PromptChunk }, stackalloc[] { 0.5f, 0f, 0f }),
            (b, q, k, v, m) => b.EmitFlashAttnExt(q, k, v, m, accPrecision: 10, rowsKind: GraphAttentionRows.PromptChunk, scale: 0.5f, softcap: 0f));
        Same((b, q, k, v, m) => b.Op(GraphOp.FlashAttnExt, stackalloc[] { q, k, v }, stackalloc long[] { 10, 4, 0, 64 }, stackalloc[] { 0.5f, 0f, 0f }),
            (b, q, k, v, m) => b.EmitFlashAttnExt(q, k, v, accPrecision: 10, kvMax: 4, band: 64, scale: 0.5f, softcap: 0f));
        // StandardDecoderModel's rotary.
        Same((b, q, k, v, m) => b.Op(GraphOp.RopeExt, stackalloc[] { q, k }, stackalloc long[] { 8, 2, 4096 }, stackalloc[] { 1e6f, 1f, 0f, 1f, 32f, 1f }),
            (b, q, k, v, m) => b.EmitRopeExt(q, k, 8, 2, 4096, 1e6f, 1f, 0f, 1f, 32f, 1f));
        Same((b, q, k, v, m) => b.Op(GraphOp.RopeExt, stackalloc[] { q, k, v }, stackalloc long[] { 8, 2, 4096 }, stackalloc[] { 1e6f, 1f, 0f, 1f, 32f, 1f }),
            (b, q, k, v, m) => b.EmitRopeExt(q, k, 8, 2, 4096, 1e6f, 1f, 0f, 1f, 32f, 1f, freqFactors: v));
        // The decoder's KV windows.
        Same((b, q, k, v, m) => b.Op(GraphOp.KvWindow, stackalloc[] { k }, stackalloc long[] { 8, 4, 2, 0, 4, 1, 1, 2 }),
            (b, q, k, v, m) => b.EmitKvWindow(k, 8, 4, 2, 0, 4, 1, queryRows: 1, gqaRatio: 2));
        Same((b, q, k, v, m) => b.Op(GraphOp.Reshape, stackalloc[] { q }, stackalloc long[] { 16, 4 }),
            (b, q, k, v, m) => b.EmitReshape(q, 16, 4));
    }

    [Fact]
    public void TypedEmittersRefuseAGapInTheirOptionalSlots()
    {
        var b = new GraphProgramBuilder(true);
        int x = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { 8, 4 }, IntPtr.Zero, 128);
        Assert.Throws<ArgumentException>(() => b.EmitReshape(x, 8, ne1: null, ne2: 4));
        Assert.Throws<ArgumentException>(() => b.EmitFlashAttnExt(x, x, x, mask: -1, sinks: x));
        // A default fills a gap in slots that have one.
        int fa = b.EmitFlashAttnExt(x, x, x, band: 64);
        Assert.Equal(new long[] { 0, 0, 0, 64 }, b.ToProgram(false, false).Entries[^1].Ip);
        Assert.True(fa > x);
    }

    [Fact]
    public void NativeBridgeIsBuiltFromTheSameSchema()
    {
        (string hash, int version) = GgmlBasicOps.NativeGraphSchema();
        Assert.Equal(GraphOpSchema.Hash, hash);
        Assert.Equal(GraphOpSchema.Version, version);
    }

    [Fact]
    public void NativeBridgeRefusesAMalformedOp()
    {
        using var c = new NetworkGraphTests.LaneContext(NetworkGraphTests.Lane.GgmlCpu);
        using var g = new GgmlGraph(64);
        var data = new float[4];
        unsafe
        {
            fixed (float* p = data)
            {
                int x = g.Leaf(GgmlLeafRole.Input, GgmlType.F32, stackalloc long[] { 4 }, (IntPtr)p, 16);
                var ex = Assert.Throws<InvalidOperationException>(() => g.Op(GgmlGraphOp.RopeExt, stackalloc[] { x, x }, stackalloc long[] { 2, 0 }));
                Assert.Contains("RopeExt takes 3 integer parameters", ex.Message);
                ex = Assert.Throws<InvalidOperationException>(() => g.Op(GgmlGraphOp.Add, stackalloc[] { x }));
                Assert.Contains("Add takes 2 operands", ex.Message);
                Assert.True(g.Op(GgmlGraphOp.Add, stackalloc[] { x, x }) > x);
            }
        }
    }
}
