// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// LTX 2.5's denoiser sharded over a tensor-parallel group (<c>--tp N</c>, DenoiserRanks): each rank holds its heads of every
/// attention and its slice of every feed-forward, sums its row-parallel partials and its whole-inner q/k norms' sums of squares
/// with the others by all-reduces, and computes the rest whole. On the managed executor of a simulated two-device group (exact
/// host all-reduce) every step equals one device's within 1e-5 relative (the partials sum in another order, and the q/k norm
/// is the explicit sum-of-squares form: rounding, no other difference; measured here at most 2.3e-7, largest error relative to
/// the largest value, and 1.8e-7 relative L2), with and without the synthetic Ingredients adapter (each rank serving its
/// slices of the factors in LoraLinear regions); on two GPUs of the CUDA tensor executor (--backend cuda --tp 2) the same
/// bound against one GPU.
/// </summary>
public sealed partial class Ltx25DenoiserTests
{
    /// <summary>The fixture checkpoint's two blocks: 14 all-reduces each (six attentions' q/k sums and to_out, two
    /// feed-forwards' down projections).</summary>
    private const int AllReducesAStep = 2 * (6 * 2 + 2);

    private const double TensorParallelBound = 1e-5;

    /// <summary>Asserts each step of <paramref name="got"/> is within <see cref="TensorParallelBound"/> of
    /// <paramref name="want"/> (relative L2 and the largest error relative to the largest value), logging both.</summary>
    private void AssertClose(IReadOnlyList<Ltx25DenoiserOutput> want, IReadOnlyList<Ltx25DenoiserOutput> got, string what,
        double bound = TensorParallelBound)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            Assert.Equal(want[i].ExecutedBlocks, got[i].ExecutedBlocks);
            var (video, videoL2) = Compare(want[i].VideoVelocity, got[i].VideoVelocity);
            var (audio, audioL2) = Compare(want[i].AudioVelocity, got[i].AudioVelocity);
            _output.WriteLine($"{what} step {i + 1}: ranks vs one device video max {video:E3} (L2 {videoL2:E3}), audio max {audio:E3} " +
                $"(L2 {audioL2:E3})");
            Assert.True(video <= bound && videoL2 <= bound && audio <= bound && audioL2 <= bound,
                $"{what} step {i + 1}: video max {video:E3} L2 {videoL2:E3}, audio max {audio:E3} L2 {audioL2:E3} past {bound:E1}");
        }
    }

    /// <summary>Every rank's outputs of the last step are rank 0's bit for bit: each rank holds the whole streams (the exact
    /// host all-reduce leaves one sum on every rank, and the rest every rank computes whole from the same inputs), so a rank
    /// that bound another input than the others (a stale table, a keyframe embedding the step does not add) shows here.</summary>
    private static void AssertRanksAgree(Ltx25Denoiser sharded, string what)
    {
        IReadOnlyList<IGraphPlan> plans = sharded.Ranks.Plans;
        for (int r = 1; r < plans.Count; r++)
            for (int o = 0; o < plans[0].Outputs.Length; o++)
                Assert.True(plans[0].Outputs[o].AsSpan().SequenceEqual(plans[r].Outputs[o]), $"{what}: rank {r}'s output {o} differs from rank 0's");
    }

    /// <summary>The blocks over a simulated two-device group (two managed executors, an exact host all-reduce): every step of
    /// every oracle case, with and without the Ingredients adapter, equals one device's within 1e-5 and the unchanged official
    /// LTX within the oracle's 2e-5; each rank compiles once a geometry (the pass's later steps and an equal input in new arrays
    /// rerun the ranks' plans); every step runs the 28 all-reduces the sharding places; each rank serves its slices of the
    /// adapter's factors in LoraLinear regions; every rank ends a step with rank 0's outputs; the partial depth, a device
    /// release and the adapter's release behave as on one device; each rank holds its share of the projections.</summary>
    [Fact]
    public void TensorParallelRanksMatchOneDeviceOnTheSimulatedGroup()
    {
        using var fixture = new Checkpoint();
        using var adapter = new FixtureAdapter(fixture.Path);
        using var group = new SimulatedDeviceGroup(2);
        using var one = new Ltx25Denoiser(fixture.Path, new CpuAllocator(BlasEnum.DotNet), adapter.Path);
        using var sharded = new Ltx25Denoiser(fixture.Path, group.GetAllocator(0), adapter.Path, tensorParallel: group);
        Assert.Null(one.Ranks);
        Assert.Null(sharded.Stages);
        Assert.Equal(2, sharded.Ranks.Count);
        Assert.True(sharded.WeightBytes < one.WeightBytes, $"a rank holds {sharded.WeightBytes} of {one.WeightBytes} projection bytes");

        // One program a rank, rerun by the pass's later steps; an equal input in new arrays emits again and finds the plans.
        int reduces = group.AllReduceCount;
        AssertClose(Pass(one, 0, adapted: false), Pass(sharded, 0, adapted: false), "case 0");
        Assert.Equal(3 * AllReducesAStep, group.AllReduceCount - reduces);
        Assert.Equal(2, sharded.Compiles);
        AssertClose(Pass(one, 0, adapted: false), Pass(sharded, 0, adapted: false), "case 0 again");
        Assert.Equal(2, sharded.Compiles);

        // The adapter: each rank binds its slices of the factors (column-parallel up rows, row-parallel down columns).
        int compiles = sharded.Compiles;
        AssertClose(Pass(one, 0, adapted: true), Pass(sharded, 0, adapted: true), "case 0 with the adapter");
        Assert.Equal(compiles + 2, sharded.Compiles);
        for (int r = 0; r < sharded.Ranks.Count; r++)
            Assert.True(((TensorGraphExecutor)sharded.Ranks.Executor(r)).LoraLinearsServedForTests > 0, $"rank {r} served no LoraLinear");

        foreach (object[] c in Cases)
        {
            int index = (int)c[0];
            var reference = Oracle.GetProperty("cases")[index];
            var first = sharded.Forward(Input(reference));
            Equal(Floats(reference, "videoVelocity"), first.VideoVelocity, 2e-5, $"case {index} video velocity over two ranks");
            Equal(Floats(reference, "audioVelocity"), first.AudioVelocity, 2e-5, $"case {index} audio velocity over two ranks");
            foreach (bool adapted in new[] { false, true })
            {
                string what = $"case {index}{(adapted ? " with the adapter" : "")}";
                AssertClose(Pass(one, index, adapted), Pass(sharded, index, adapted), what);
                AssertRanksAgree(sharded, what);
            }
        }

        // Partial depth: the first block, then the heads.
        var input = Input(Oracle.GetProperty("cases")[1]);
        AssertClose(new[] { one.Forward(input, 1) }, new[] { sharded.Forward(input, 1) }, "one block");

        // A release before the decode, and the adapter released as a set replacing it is: the next steps upload again.
        sharded.ReleaseDevice();
        one.ReleaseDevice();
        AssertClose(Pass(one, 2, adapted: true), Pass(sharded, 2, adapted: true), "case 2 after the device release");
        sharded.ReleaseLora();
        one.ReleaseLora();
        AssertClose(Pass(one, 3, adapted: false), Pass(sharded, 3, adapted: false), "case 3 after the adapter release");
        AssertRanksAgree(sharded, "case 3 after the adapter release");
    }

    /// <summary>A degree the fixture's two heads do not divide is refused before any device is touched, and so are a group
    /// whose first device is not the denoiser's allocator and a split with it (tensor-parallel stages are not implemented).
    /// One device is the unsharded denoiser.</summary>
    [Fact]
    public void TensorParallelRefusesDegreesThatDoNotDivideAndTheHybrid()
    {
        using var fixture = new Checkpoint();
        foreach (int degree in new[] { 3, 4 })
        {
            using var group = new SimulatedDeviceGroup(degree);
            var refused = Assert.Throws<NotSupportedException>(() => new Ltx25Denoiser(fixture.Path, group.GetAllocator(0), tensorParallel: group));
            Assert.Contains("heads", refused.Message);
            Assert.Contains($"--tp {degree}", refused.Message);
        }
        using var two = new SimulatedDeviceGroup(2);
        Assert.Throws<ArgumentException>(() => new Ltx25Denoiser(fixture.Path, new CpuAllocator(BlasEnum.DotNet), tensorParallel: two));
        var hybrid = Assert.Throws<NotSupportedException>(() => new Ltx25Denoiser(fixture.Path, two.GetAllocator(0),
            devices: new[] { two.GetAllocator(0), new CpuAllocator(BlasEnum.DotNet) }, tensorParallel: two));
        Assert.Contains("not both", hybrid.Message);
        using var single = new SimulatedDeviceGroup(1);
        using var whole = new Ltx25Denoiser(fixture.Path, single.GetAllocator(0), tensorParallel: single);
        Assert.Null(whole.Ranks);
    }

    /// <summary>The shapes of one LTX 2.5 block of the released checkpoint (video 4096 wide, audio 2048, 32 heads each, 4x
    /// feed-forwards), each projection in <paramref name="type"/>.</summary>
    private static IEnumerable<(string, GgmlTensorType, long, long)> ReleasedBlock(GgmlTensorType type)
    {
        const long v = 4096, a = 2048, heads = 32;
        var shapes = new List<(string, long, long)>();
        void Attention(string name, long query, long context, long inner)
        {
            shapes.Add((name + ".to_q", query, inner));
            shapes.Add((name + ".to_k", context, inner));
            shapes.Add((name + ".to_v", context, inner));
            shapes.Add((name + ".to_gate_logits", query, heads));
            shapes.Add((name + ".to_out.0", inner, query));
        }
        const string b = "transformer_blocks.0.";
        Attention(b + "attn1", v, v, v);
        Attention(b + "attn2", v, v, v);
        Attention(b + "audio_attn1", a, a, a);
        Attention(b + "audio_attn2", a, a, a);
        Attention(b + "audio_to_video_attn", v, a, a);
        Attention(b + "video_to_audio_attn", a, v, a);
        shapes.Add((b + "ff.net.0.proj", v, 4 * v));
        shapes.Add((b + "ff.net.2", 4 * v, v));
        shapes.Add((b + "audio_ff.net.0.proj", a, 4 * a));
        shapes.Add((b + "audio_ff.net.2", 4 * a, a));
        shapes.Add(("adaln_single.linear", v, 9 * v));
        shapes.Add(("proj_out", v, 128));
        return shapes.Select(s => (s.Item1, type, s.Item2, s.Item3));
    }

    /// <summary>On the released checkpoint's shapes: a degree that does not divide the 32 heads is refused; a Q4_K checkpoint
    /// splits over up to eight GPUs (a 2,048-wide row-parallel projection is eight 256-value blocks a row) and is refused at
    /// sixteen, where a rank's share of a row would start inside a block; a BF16 one (one value a block) splits over 16 and 32.
    /// Replicated projections (the AdaLN MLPs, the heads) are never cut.</summary>
    [Fact]
    public void TensorParallelRefusalFollowsTheHeadsAndTheRowParallelQuantBlocks()
    {
        foreach (int degree in new[] { 1, 2, 4, 8 })
            Assert.Null(Ltx25DenoiserWeights.TensorParallelRefusal(32, 32, ReleasedBlock(GgmlTensorType.Q4_K), degree));
        string blocks = Ltx25DenoiserWeights.TensorParallelRefusal(32, 32, ReleasedBlock(GgmlTensorType.Q4_K), 16);
        Assert.NotNull(blocks);
        Assert.Contains("quant block", blocks);
        Assert.Contains("to_out.0", blocks);
        foreach (int degree in new[] { 16, 32 })
            Assert.Null(Ltx25DenoiserWeights.TensorParallelRefusal(32, 32, ReleasedBlock(GgmlTensorType.BF16), degree));
        foreach (int degree in new[] { 3, 64 })
            Assert.Contains("heads", Ltx25DenoiserWeights.TensorParallelRefusal(32, 32, ReleasedBlock(GgmlTensorType.BF16), degree));
    }

    /// <summary>The family's contract: tensor parallelism and layer split on --backend cuda (the CUDA tensor executor's
    /// devices); ggml_cuda refused with the advice to use cuda, a single-device backend refused, a distributed group and the
    /// hybrid (both at once) refused.</summary>
    [Fact]
    public void TensorParallelismIsTheContractsOnCudaAndTheHybridAGap()
    {
        Assert.True(ModelArchitectureRegistry.TryGet(Ltx25Model.ArchitectureId, out var arch));
        Assert.Equal(new[] { BackendType.Cuda }, arch.Serving.TensorParallel.Backends);
        Assert.Empty(arch.Serving.HybridParallel.Backends);
        Assert.False(arch.SupportsDistributedTensorParallel);

        ITensorParallelGroup group = null;
        Assert.Equal(2, ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cuda, 2, ref group, out int split));
        Assert.Equal(1, split);
        Assert.Null(group);
        var ggml = Assert.Throws<NotSupportedException>(() => ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCuda, 2, ref group, out _));
        Assert.Contains("--backend cuda", ggml.Message);
        Assert.Contains("--tp 2", ggml.Message);
        var cpu = Assert.Throws<NotSupportedException>(() => ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cpu, 2, ref group, out _));
        Assert.Contains("multi-device", cpu.Message);
        var hybrid = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cuda, 2, ref group, out _, requestedLayerSplitDegree: 2));
        Assert.Contains("ltx-2.5", hybrid.Message);
        ITensorParallelGroup distributed = new DistributedStub();
        var nodes = Assert.Throws<NotSupportedException>(() => ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cuda, 2, ref distributed, out _));
        Assert.Contains("--tp-node-id/--tp-peers", nodes.Message);
    }

    /// <summary>The blocks over two GPUs of the CUDA tensor executor (TensorSharp.Cuda.TensorParallelGroup, as --backend cuda
    /// --tp 2 makes it): every step of every oracle case equals one GPU's within 1e-5 (each rank runs one GPU's kernels on its
    /// shards; the partials sum on the devices), with and without the Ingredients adapter.</summary>
    [MultiCudaTheory(2)]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorParallelRanksOnTwoGpusMatchOneGpu(bool adapted)
    {
        using var fixture = new Checkpoint();
        using var adapter = new FixtureAdapter(fixture.Path);
        var want = new List<List<Ltx25DenoiserOutput>>();
        using (var first = new CudaAllocator(0))
        using (var one = new Ltx25Denoiser(fixture.Path, first, adapter.Path))
            foreach (object[] c in Cases) want.Add(Pass(one, (int)c[0], adapted));
        using var group = new TensorSharp.Cuda.TensorParallelGroup(2);
        using var sharded = new Ltx25Denoiser(fixture.Path, group.GetAllocator(0), adapter.Path, tensorParallel: group);
        Assert.Equal(2, sharded.Ranks.Count);
        Assert.IsType<CudaGraphExecutor>(sharded.Ranks.Executor(1));
        Assert.True(((TensorGraphExecutor)sharded.Ranks.Executor(1)).DirectProducts);
        foreach (object[] c in Cases)
            AssertClose(want[(int)c[0]], Pass(sharded, (int)c[0], adapted), $"case {c[0]}{(adapted ? " with the adapter" : "")} on two GPUs");
    }

    /// <summary>A live group over two nodes: the gate only reads whether one exists.</summary>
    private sealed class DistributedStub : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 4;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }
}
