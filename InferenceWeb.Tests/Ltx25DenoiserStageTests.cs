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
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// LTX 2.5's denoiser over several devices as whole-block stages (<c>--layer-split N</c>, DenoiserStages): each stage's program
/// runs its blocks on its own device and hands the video and audio residual streams on as F32 rows. On the managed executor of
/// two host "devices" every step equals one device's bit for bit (the oracle checkpoint's two blocks, one a stage), with and
/// without the synthetic Ingredients adapter over both blocks (each stage binding its own block's factors); on two GPUs of the
/// CUDA tensor executor (--backend cuda) every step stays within 1e-5 of one GPU's.
/// </summary>
public sealed partial class Ltx25DenoiserTests
{
    private readonly ITestOutputHelper _output;

    public Ltx25DenoiserTests(ITestOutputHelper output) => _output = output;

    /// <summary>A pass of three steps on <paramref name="model"/>: the oracle case's step, then two later sigmas over the same
    /// arrays (the sampler steps its state in place, so each stage reruns its kept trace with new timestep features).</summary>
    private static List<Ltx25DenoiserOutput> Pass(Ltx25Denoiser model, int index, bool adapted)
    {
        var input = Input(Oracle.GetProperty("cases")[index], adapted);
        var outputs = new List<Ltx25DenoiserOutput> { model.Forward(input) };
        foreach (float factor in new[] { .5f, .25f })
            outputs.Add(model.Forward(At(input, input.VideoSigma * factor, input.AudioSigma * factor)));
        return outputs;
    }

    private static void AssertBitExact(IReadOnlyList<Ltx25DenoiserOutput> want, IReadOnlyList<Ltx25DenoiserOutput> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            Assert.Equal(want[i].ExecutedBlocks, got[i].ExecutedBlocks);
            Assert.True(want[i].VideoVelocity.Select(BitConverter.SingleToInt32Bits).SequenceEqual(got[i].VideoVelocity.Select(BitConverter.SingleToInt32Bits)),
                $"{what}: step {i + 1}'s video velocity differs from one device's");
            Assert.True(want[i].AudioVelocity.Select(BitConverter.SingleToInt32Bits).SequenceEqual(got[i].AudioVelocity.Select(BitConverter.SingleToInt32Bits)),
                $"{what}: step {i + 1}'s audio velocity differs from one device's");
        }
    }

    /// <summary>The blocks over two host devices (two managed executors): every step of every oracle case, with and without the
    /// Ingredients adapter, equals one device's bit for bit; each stage compiles once a geometry (the pass's later steps and an
    /// equal input in new arrays rerun its plan); the partial depth, a device release and the adapter's release behave as on
    /// one device; each device holds its stage's share of the projections.</summary>
    [Fact]
    public void WholeBlockStagesMatchOneDeviceOnTheManagedExecutor()
    {
        using var fixture = new Checkpoint();
        using var adapter = new FixtureAdapter(fixture.Path);
        var host = new CpuAllocator(BlasEnum.DotNet);
        var devices = new IAllocator[] { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
        using var one = new Ltx25Denoiser(fixture.Path, host, adapter.Path);
        using var staged = new Ltx25Denoiser(fixture.Path, devices[0], adapter.Path, devices);
        Assert.Null(one.Stages);
        Assert.Equal(2, staged.Stages.Count);
        Assert.Equal((0, 1), (staged.Stages.First(0), staged.Stages.End(0)));
        Assert.Equal((1, 2), (staged.Stages.First(1), staged.Stages.End(1)));
        Assert.True(staged.WeightBytes < one.WeightBytes, $"a stage holds {staged.WeightBytes} of {one.WeightBytes} projection bytes");

        // One program a stage, rerun by the pass's later steps; an equal input in new arrays emits again and finds the plans.
        AssertBitExact(Pass(one, 0, adapted: false), Pass(staged, 0, adapted: false), "case 0");
        Assert.Equal(2, staged.Compiles);
        AssertBitExact(Pass(one, 0, adapted: false), Pass(staged, 0, adapted: false), "case 0 again");
        Assert.Equal(2, staged.Compiles);

        // The adapter: each stage binds its own block's factors (both stages add the update in place, as LoraLinear regions).
        int compiles = staged.Compiles;
        AssertBitExact(Pass(one, 0, adapted: true), Pass(staged, 0, adapted: true), "case 0 with the adapter");
        Assert.Equal(compiles + 2, staged.Compiles);
        for (int s = 0; s < staged.Stages.Count; s++)
            Assert.True(((TensorGraphExecutor)staged.Stages.Executor(s)).LoraLinearsServedForTests > 0, $"stage {s} served no LoraLinear");

        foreach (object[] c in Cases)
            foreach (bool adapted in new[] { false, true })
                AssertBitExact(Pass(one, (int)c[0], adapted), Pass(staged, (int)c[0], adapted), $"case {c[0]}{(adapted ? " with the adapter" : "")}");

        // Partial depth: the first stage's block, then the last stage's heads.
        var input = Input(Oracle.GetProperty("cases")[1]);
        AssertBitExact(new[] { one.Forward(input, 1) }, new[] { staged.Forward(input, 1) }, "one block");

        // A release before the decode, and the adapter released as a set replacing it is: the next steps upload again.
        staged.ReleaseDevice();
        one.ReleaseDevice();
        AssertBitExact(Pass(one, 2, adapted: true), Pass(staged, 2, adapted: true), "case 2 after the device release");
        staged.ReleaseLora();
        one.ReleaseLora();
        AssertBitExact(Pass(one, 3, adapted: false), Pass(staged, 3, adapted: false), "case 3 after the adapter release");
    }

    /// <summary>A split's first device is the denoiser's own allocator (the latents and the keyframe embedding live there), and
    /// a split needs a block a device.</summary>
    [Fact]
    public void WholeBlockStagesRefuseAForeignFirstDeviceAndMoreDevicesThanBlocks()
    {
        using var fixture = new Checkpoint();
        var host = new CpuAllocator(BlasEnum.DotNet);
        Assert.Throws<ArgumentException>(() => new Ltx25Denoiser(fixture.Path, host, devices: new IAllocator[]
            { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) }));
        var three = new IAllocator[] { host, new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
        Assert.Contains("a block a device", Assert.Throws<NotSupportedException>(() => new Ltx25Denoiser(fixture.Path, host, devices: three)).Message);
        // One device is the unsplit denoiser.
        using var single = new Ltx25Denoiser(fixture.Path, host, devices: new IAllocator[] { host });
        Assert.Null(single.Stages);
    }

    /// <summary>The family's contract: it is on the shared diffusion runtime; --layer-split N is honoured on --backend cuda (the
    /// CUDA tensor executor's devices) and refused on ggml_cuda with the advice to use cuda, and on a single-device backend
    /// (tensor parallelism: <see cref="TensorParallelismIsTheContractsOnCudaAndTheHybridAGap"/>).</summary>
    [Fact]
    public void LayerSplitIsTheContractsOnCuda()
    {
        Assert.True(ModelArchitectureRegistry.TryGet(Ltx25Model.ArchitectureId, out var arch));
        Assert.Equal(ServingImplementation.SharedDiffusionRuntime, arch.Serving.Implementation);
        Assert.Equal(new[] { BackendType.Cuda }, arch.Serving.LayerSplit.Backends);
        Assert.Empty(arch.Serving.HybridParallel.Backends);

        ITensorParallelGroup group = null;
        Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cuda, 1, ref group, out int split, requestedLayerSplitDegree: 2));
        Assert.Equal(2, split);
        var ggml = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCuda, 1, ref group, out _, requestedLayerSplitDegree: 2));
        Assert.Contains("--backend cuda", ggml.Message);
        var cpu = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.Cpu, 1, ref group, out _, requestedLayerSplitDegree: 2));
        Assert.Contains("multi-device", cpu.Message);
    }

    /// <summary>The blocks over two GPUs of the CUDA tensor executor (two CudaAllocators, as --backend cuda --layer-split 2 makes
    /// them): every step of every oracle case equals one GPU's within 1e-5 relative (the stages run one GPU's kernels; only the
    /// streams cross, as F32 rows), with and without the Ingredients adapter.</summary>
    [MultiCudaTheory(2)]
    [InlineData(false)]
    [InlineData(true)]
    public void WholeBlockStagesOnTwoGpusMatchOneGpu(bool adapted)
    {
        using var fixture = new Checkpoint();
        using var adapter = new FixtureAdapter(fixture.Path);
        using var first = new CudaAllocator(0);
        using var second = new CudaAllocator(1);
        var want = new List<List<Ltx25DenoiserOutput>>();
        using (var one = new Ltx25Denoiser(fixture.Path, first, adapter.Path))
            foreach (object[] c in Cases) want.Add(Pass(one, (int)c[0], adapted));
        using var staged = new Ltx25Denoiser(fixture.Path, first, adapter.Path, new IAllocator[] { first, second });
        Assert.Equal(2, staged.Stages.Count);
        Assert.IsType<CudaGraphExecutor>(staged.Stages.Executor(1));
        Assert.True(((TensorGraphExecutor)staged.Stages.Executor(1)).DirectProducts);
        foreach (object[] c in Cases)
        {
            int index = (int)c[0];
            List<Ltx25DenoiserOutput> got = Pass(staged, index, adapted);
            for (int i = 0; i < got.Count; i++)
            {
                var (video, videoL2) = Compare(want[index][i].VideoVelocity, got[i].VideoVelocity);
                var (audio, audioL2) = Compare(want[index][i].AudioVelocity, got[i].AudioVelocity);
                _output.WriteLine($"case {index} step {i + 1}{(adapted ? " with the adapter" : "")}: stages vs one GPU video max {video:E3} " +
                    $"(L2 {videoL2:E3}), audio max {audio:E3} (L2 {audioL2:E3})");
                Assert.True(video <= 1e-5 && audio <= 1e-5, $"case {index} step {i + 1}: video {video:E3}, audio {audio:E3}");
            }
        }
    }

    /// <summary>The largest error relative to the largest magnitude, and the relative L2 error, of <paramref name="got"/>.</summary>
    private static (double Max, double L2) Compare(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        double maxError = 0, maxValue = 0, error2 = 0, value2 = 0;
        for (int i = 0; i < want.Length; i++)
        {
            Assert.True(float.IsFinite(got[i]));
            double e = (double)got[i] - want[i];
            maxError = Math.Max(maxError, Math.Abs(e));
            maxValue = Math.Max(maxValue, Math.Abs(want[i]));
            error2 += e * e;
            value2 += (double)want[i] * want[i];
        }
        return (maxError / Math.Max(maxValue, 1e-30), Math.Sqrt(error2 / Math.Max(value2, 1e-30)));
    }
}
