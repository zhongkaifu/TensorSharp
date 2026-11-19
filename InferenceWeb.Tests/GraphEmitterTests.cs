// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The shared GraphEmitter that the decoder runtime's GraphStep and the diffusion runtime emit programs through: it
// records exactly what a GraphProgramBuilder records (no hidden nodes, the same signature), keeps its plain inputs'
// bindings in order, and its programs run on the tensor executor and on the lane's GGML backend.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;

namespace InferenceWeb.Tests;

public sealed class GraphEmitterTests
{
    private const int K = 64, M = 24, N = 5;
    private const float Eps = 1e-6f;
    private static readonly IAllocator Cpu = new CpuAllocator(BlasEnum.DotNet);

    private sealed class ManagedExecutor : TensorGraphExecutor
    {
        public ManagedExecutor() : base(Cpu) { }
        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x) => throw new NotSupportedException();
    }

    private static float[] Values(int n, int salt) => Enumerable.Range(0, n).Select(i => ((i * 37 + salt * 11) % 29 - 14) / 16f).ToArray();

    private static Tensor TensorOf(IAllocator allocator, float[] values, params long[] sizes)
    {
        var t = new Tensor(allocator, DType.Float32, sizes);
        t.SetElementsAsFloat(values);
        return t;
    }

    /// <summary>y = w · (rms_norm(x) * g), with a strided view of x's first half copied out as a second output.</summary>
    private static (int Y, int Half) Emit(GraphEmitter e, Tensor w, Tensor g, float[] x)
    {
        int input = e.InputF32(x, K, N);
        int normed = e.RmsNorm(input, e.TensorWeight(g, K, 1), Eps);
        int y = e.Op(GraphOp.MulMat, e.TensorWeight(w), normed);
        int half = e.Cont(e.View2(input, K / 2, N, K * 4L, 0));
        return (e.Output(y, (long)M * N), e.Output(half, (long)K / 2 * N));
    }

    private static float[] Expected(float[] w, float[] g, float[] x)
    {
        var y = new float[M * N];
        for (int n = 0; n < N; n++)
        {
            double sum = 0;
            for (int k = 0; k < K; k++) sum += (double)x[n * K + k] * x[n * K + k];
            float scale = (float)(1.0 / Math.Sqrt(sum / K + Eps));
            for (int m = 0; m < M; m++)
            {
                double dot = 0;
                for (int k = 0; k < K; k++) dot += (double)w[m * K + k] * (x[n * K + k] * scale * g[k]);
                y[n * M + m] = (float)dot;
            }
        }
        return y;
    }

    [Fact]
    public void TheEmitterRecordsWhatABuilderRecords()
    {
        var executor = new ManagedExecutor();
        float[] wv = Values(M * K, 1), gv = Values(K, 2), xv = Values(K * N, 3);
        using Tensor w = TensorOf(Cpu, wv, M, K), g = TensorOf(Cpu, gv, K);
        using var emitter = new GraphEmitter(executor);
        Emit(emitter, w, g, xv);
        GraphProgram emitted = emitter.ToProgram(replayable: true);

        // The same program by hand: leaves bound by the tensors' identities (this executor binds objects).
        var b = new GraphProgramBuilder(record: true);
        int input = b.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { K, N }, IntPtr.Zero, 0);
        int gl = b.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { K, 1 }, GraphIdentity.Of(g.Storage, g.StorageOffset), g.Storage.ByteLength, null, g);
        int normed = b.Op(GraphOp.Mul, b.Op(GraphOp.RmsNorm, stackalloc[] { input }, default, stackalloc[] { Eps }), gl);
        int wl = b.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { K, M }, GraphIdentity.Of(w.Storage, w.StorageOffset), w.Storage.ByteLength, null, w);
        int y = b.Op(GraphOp.MulMat, wl, normed);
        int view = b.Op(GraphOp.View, stackalloc[] { input }, stackalloc long[] { 2, K / 2, N, 1, 1, K * 4L, 0, 0, 0 });
        int half = b.Op(GraphOp.Cont, view);
        b.Output(y, (long)M * N);
        b.Output(half, (long)K / 2 * N);
        GraphProgram built = b.ToProgram(replayable: true, reorder: false);

        Assert.Equal(built.Signature, emitted.Signature);
        Assert.Equal(built.Signature, emitter.Signature);
        Assert.Equal(built.Entries.Count, emitted.Entries.Count);
        for (int i = 0; i < built.Entries.Count; i++)
        {
            GraphNode a = built.Entries[i], c = emitted.Entries[i];
            Assert.Equal((a.Kind, a.Op, a.Role, a.Format, a.Data, a.Bytes, a.Node), (c.Kind, c.Op, c.Role, c.Format, c.Data, c.Bytes, c.Node));
            Assert.Equal(a.Srcs ?? Array.Empty<int>(), c.Srcs ?? Array.Empty<int>());
            Assert.Equal(a.Ip ?? Array.Empty<long>(), c.Ip ?? Array.Empty<long>());
            Assert.Equal(a.Fp ?? Array.Empty<float>(), c.Fp ?? Array.Empty<float>());
            Assert.Equal(a.Ne ?? Array.Empty<long>(), c.Ne ?? Array.Empty<long>());
            Assert.Same(a.Handle, c.Handle);
        }
        // A contiguous node is its own Cont: no copy is recorded.
        using var plain = new GraphEmitter(executor, record: false);
        int x = plain.InputF32(xv, K, N);
        Assert.Equal(x, plain.Cont(x));
    }

    [Fact]
    public void PlainInputsAreBoundInOrderAndAttentionCarriesItsRowMarks()
    {
        using var e = new GraphEmitter(new ManagedExecutor());
        int a = e.InputF32(new float[8], 8);
        int b = e.InputI32(new int[4], 4);
        int c = e.InputF16(new ushort[6], 6);
        GraphRunInputs inputs = e.RunInputs();
        Assert.Equal(new[] { a, b, c }, inputs.Nodes);
        Assert.Equal(new long[] { 32, 16, 12 }, inputs.Bytes);
        Assert.All(inputs.Data, p => Assert.NotEqual(IntPtr.Zero, p));

        int q = e.InputF32(new float[64 * 4], 64, 4, 1), kv = e.InputF32(new float[64 * 4], 64, 4, 1);
        int plainNode = e.FlashAttention(q, kv, kv, -1, 0.125f);
        int windowNode = e.FlashAttention(q, kv, kv, -1, 0.125f, kvMax: 3);
        int chunkNode = e.FlashAttention(q, kv, kv, -1, 0.125f, promptChunk: true);
        GraphProgram program = e.ToProgram(replayable: false);
        GraphNode Node(int node) => program.Entries[IndexOfNode(program, node)];
        Assert.Equal(new long[] { 10 }, Node(plainNode).Ip);
        Assert.Equal(new long[] { 10, 3 }, Node(windowNode).Ip);
        Assert.Equal(new long[] { 10, 0, GraphAttentionRows.PromptChunk }, Node(chunkNode).Ip);
        Assert.Equal(GraphOp.FlashAttnExt, Node(chunkNode).Op);
        Assert.Equal(new[] { q, kv, kv }, Node(plainNode).Srcs);
    }

    /// <summary>The entry index of node <paramref name="node"/> (entries that are not nodes have no index).</summary>
    private static int IndexOfNode(GraphProgram program, int node)
    {
        int seen = -1;
        for (int i = 0; i < program.Entries.Count; i++)
        {
            if (program.Entries[i].Kind is GraphNodeKind.Leaf or GraphNodeKind.Op) seen++;
            if (seen == node) return i;
        }
        return -1;
    }

    [Fact]
    public void AnEmittedProgramRunsOnTheTensorExecutor()
    {
        float[] wv = Values(M * K, 4), gv = Values(K, 5), xv = Values(K * N, 6);
        using Tensor w = TensorOf(Cpu, wv, M, K), g = TensorOf(Cpu, gv, K);
        using var e = new GraphEmitter(new ManagedExecutor());
        var (y, half) = Emit(e, w, g, xv);
        using IGraphPlan plan = e.Compile(replayable: true);
        Assert.True(e.Run(plan), plan.LastError);
        Check(plan, y, half, wv, gv, xv, 1e-5f);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void AnEmittedProgramRunsOnGgmlCpu() => RunOnGgml(GgmlBackendType.Cpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void AnEmittedProgramRunsOnGgmlMetal() => RunOnGgml(GgmlBackendType.Metal);

    private static void RunOnGgml(GgmlBackendType backend)
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], backend), 0);
        IGraphExecutor executor = GraphExecutors.For(allocator);
        Assert.True(executor.Capabilities.BindsHostPointers);
        float[] wv = Values(M * K, 7), gv = Values(K, 8), xv = Values(K * N, 9);
        using Tensor w = TensorOf(allocator, wv, M, K), g = TensorOf(allocator, gv, K);
        using var e = new GraphEmitter(executor);
        var (y, half) = Emit(e, w, g, xv);
        using IGraphPlan plan = e.Compile(replayable: true);
        Assert.True(e.Run(plan), plan.LastError);
        Check(plan, y, half, wv, gv, xv, 2e-4f);
        // A second run of the replayable plan with new input values reads them.
        float[] xv2 = Values(K * N, 10);
        using var again = new GraphEmitter(executor);
        Emit(again, w, g, xv2);
        Assert.Equal(e.Signature, again.Signature);
        Assert.True(again.Run(plan), plan.LastError);
        Check(plan, y, half, wv, gv, xv2, 2e-4f);
    }

    private static void Check(IGraphPlan plan, int y, int half, float[] w, float[] g, float[] x, float tolerance)
    {
        float[] expected = Expected(w, g, x);
        Assert.Equal(expected.Length, plan.Outputs[y].Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - plan.Outputs[y][i]) <= tolerance * (1 + Math.Abs(expected[i])),
                $"y[{i}] = {plan.Outputs[y][i]}, expected {expected[i]}");
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K / 2; k++)
                Assert.Equal(x[n * K + k], plan.Outputs[half][n * (K / 2) + k]);
    }
}
