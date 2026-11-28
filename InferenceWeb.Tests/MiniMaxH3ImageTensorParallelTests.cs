// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using G = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Grammar;
using P = InferenceWeb.Tests.MiniMaxH3ImageLoraFixture.Proj;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image on direct CUDA (<see cref="BackendType.Cuda"/>: the CUDA tensor executor) and its denoiser sharded over a
/// tensor-parallel group (<c>--tp N</c>, <see cref="DenoiserRanks"/>: each rank its heads of every attention and its slice of
/// every MLP, an all-reduce after the attention output and after fc2), on the synthetic checkpoint
/// (<see cref="MiniMaxH3ImageFixture"/>: 4 heads of 64, FFN 512, 2 blocks and a token-refiner block).
/// <para>
/// What runs here: the sharding on the managed executor over a simulated group of two host devices
/// (<see cref="SimulatedDeviceGroup"/>: its exact host sum), every step against one device's, with LoRA adapters (their
/// column- and row-parallel slices) and after them; the refusals; the decoder bound as a tensor executor binds it (the
/// managed executor). What needs a GPU (skipped without one; bounds marked provisional until measured there): direct CUDA's
/// denoiser against the managed executor, its whole-block stages and its sharded steps on two GPUs against one, its device
/// copies released, its decoder against the frozen legacy pixels, and the model's load with <c>--tp 2</c>.
/// </para>
/// </summary>
public sealed class MiniMaxH3ImageTensorParallelTests(ITestOutputHelper output) : IDisposable
{
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    /// <summary>The managed sharded steps against one device, where the products keep F32 activations (F32 and BF16
    /// projections): measured within 4e-7 (relative L2) and 2.3e-6 (largest difference over the largest value), with and
    /// without adapters. The two partials' sum rounds otherwise than the one product only in the last bits.</summary>
    private const double ManagedRelL2 = 1e-5, ManagedMax = 1e-4;

    /// <summary>Q8_0 projections on the managed executor quantize a product's activations to 8 bits (ManagedQuantizedOps): a
    /// last-bit difference of the stream after a block's all-reduce can flip one rounding of the next block's quantized rows,
    /// as MiniMax-H3 video's managed Q8_0 lane allows for. Measured: without adapters within 1.2e-7 (L2) and 2.4e-7 (max);
    /// with them one step at 2.8e-5 (L2) and 2.9e-4 (max).</summary>
    private const double QuantizedRelL2 = 1e-4, QuantizedMax = 1e-3;

    private readonly MiniMaxH3ImageLoraFixture _fx = new();

    public void Dispose() => _fx.Dispose();

    private static float[] Random(Random rng, long n, float scale = 1f)
    {
        var v = new float[n];
        for (long i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>Text to image at 272 x 272 (289 image tokens), an edit's packed sequence (two references, mixed text and
    /// vision rows: several modulation segments), and 1024 x 1024 (4096 tokens).</summary>
    private static MiniMaxH3ImageGoldens.Request Geometry(string name) => name switch
    {
        "text" => MiniMaxH3ImageGoldens.Request.TextToImage,
        "edit" => MiniMaxH3ImageGoldens.Request.WithReferences,
        "wide" => MiniMaxH3ImageGoldens.Request.Wide,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    /// <summary>Two Euler steps of a speed recipe from one seed's noise: each step's velocity, the state advanced with
    /// <paramref name="advanceWith"/>'s (the reference's) so both denoisers see the same inputs.</summary>
    private static List<float[]> Steps(MiniMaxH3ImageFamily family, IDenoiser denoiser, AdapterSet adapters,
        MiniMaxH3ImageGoldens.Request rq, List<float[]> advanceWith = null, int seed = 11)
    {
        var rng = new Random(seed);
        var text = new ConditioningSet(Random(rng, (long)rq.TextRows * MiniMaxH3ImageFixture.TextDim, 2f), rq.TextRows,
            MiniMaxH3ImageFixture.TextDim, rq.Modalities);
        LatentRef[] references = (rq.References ?? Array.Empty<(int, int)>()).Select((r, i) =>
        {
            var shape = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, r.H, r.W);
            return new LatentRef(i, shape, family.ToRows(new LatentTensor(shape, Random(rng, shape.Count))));
        }).ToArray();
        var target = new LatentShape(MiniMaxH3ImageFixture.Channels, 1, rq.LatentHeight, rq.LatentWidth);
        float[] x = family.ToRows(new LatentTensor(target, family.Noise(seed).Next(target.Count)));
        var contract = new SamplingContract { Name = "test", Sigmas = new[] { 1f, 12f / 13f, 0f }, Heads = HeadPolicy.ByStep, MinGenerationArea = 1024 };
        SamplingRecipe recipe = contract.Apply(family.DefaultRecipe);
        float[] sigmas = recipe.SigmasFor(recipe.Steps);
        var result = new List<float[]>();
        for (int i = 0; i < recipe.Steps; i++)
        {
            var (start, stop) = recipe.HeadsFor(i, sigmas[i], sigmas[i + 1]);
            var state = new StepState(i, recipe.Steps, sigmas[i], sigmas[i + 1], start, stop);
            var inputs = new DenoiseStepInputs
            {
                Latent = x, Conditioning = text, References = references,
                Layout = family.Pack(text, references, target, state), Step = state, Recipe = recipe,
            };
            float[] v = (float[])denoiser.Step(inputs, adapters, CancellationToken.None).Velocity.Clone();
            result.Add(v);
            float[] advance = advanceWith?[i] ?? v;
            for (long k = 0; k < x.LongLength; k++) x[k] += advance[k] * (sigmas[i + 1] - sigmas[i]);
        }
        return result;
    }

    private static (double Max, double RelL2) Compare(float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        double scale = want.Max(v => Math.Abs((double)v)), worst = 0, l2 = 0, n2 = 0;
        for (int k = 0; k < want.Length; k++)
        {
            Assert.True(float.IsFinite(got[k]), $"value {k} is {got[k]}");
            double d = got[k] - (double)want[k];
            worst = Math.Max(worst, Math.Abs(d));
            l2 += d * d;
            n2 += (double)want[k] * want[k];
        }
        return (worst / scale, Math.Sqrt(l2 / n2));
    }

    /// <summary>Every step of <paramref name="got"/> within the bounds of <paramref name="want"/>'s, each logged.</summary>
    private void AssertClose(List<float[]> want, List<float[]> got, string what, double relL2Bound, double maxBound)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            var (max, relL2) = Compare(want[i], got[i]);
            output.WriteLine($"{what} step {i + 1}: max {max:E3}, L2 {relL2:E3}");
            Assert.True(relL2 <= relL2Bound && max <= maxBound,
                $"{what} step {i + 1}: max {max:E3} (bound {maxBound:E1}), L2 {relL2:E3} (bound {relL2Bound:E1})");
        }
    }

    private static void AssertBitExact(List<float[]> want, List<float[]> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            int differ = 0;
            for (int k = 0; k < want[i].Length; k++)
                if (BitConverter.SingleToInt32Bits(want[i][k]) != BitConverter.SingleToInt32Bits(got[i][k])) differ++;
            Assert.True(differ == 0, $"{what} step {i + 1}: {differ} of {want[i].Length} values differ");
        }
    }

    /// <summary>A diffusers speed adapter on every projection of every block (q, k and v apart: three slots a rank's up rows
    /// are a view of; the refiner too, whole on every rank) and a kohya style adapter at strength 0.6 on the dense fused qkv
    /// and fc1 (one slot spanning the stacked parts: a rank's up rows are a copy of its run of each), trunk only.</summary>
    private List<LoraSpec> Adapters(int seed = 21)
    {
        var speed = MiniMaxH3ImageLoraFixture.Random(seed, 8, 0.5f, perThirdQkv: true, magnitude: 1f);
        var style = MiniMaxH3ImageLoraFixture.Random(seed + 1, 4, 1f, perThirdQkv: false, magnitude: 0.8f, trunkOnly: true,
            projections: new[] { P.Qkv, P.Fc1 });
        string a = _fx.Write(MiniMaxH3ImageLoraFixture.Build(speed, G.Diffusers, prefix: "transformer."), $"tp-speed-{seed}.safetensors");
        string b = _fx.Write(MiniMaxH3ImageLoraFixture.Build(style, G.Kohya), $"tp-style-{seed}.safetensors");
        string config = _fx.WriteConfig($"tp-speed-{seed}.json", a, name: "Speed", steps: 2);
        return new List<LoraSpec> { new(config), new(b, 0.6f) };
    }

    // ================================================================ tensor parallelism on the managed executor

    /// <summary>
    /// The denoiser over a simulated group of two host devices, on the managed executor each (a rank's program its two heads
    /// and 256 MLP columns, the rest whole): every step's velocity equals one device's within <see cref="ManagedRelL2"/> on
    /// text to image and on an edit's packed sequence (<see cref="QuantizedRelL2"/> where the managed products quantize their
    /// activations), with LoRA adapters (every projection's slices) and after them; each rank compiles once a geometry and
    /// set, and every block sums its two row-parallel partials.
    /// </summary>
    [Theory]
    [InlineData(F32)]
    [InlineData(BF16)]
    [InlineData(Q8_0)]
    public void ShardedStepsMatchOneDeviceOnTheManagedExecutor(int format)
    {
        bool quantized = format == Q8_0;
        double l2Bound = quantized ? QuantizedRelL2 : ManagedRelL2, maxBound = quantized ? QuantizedMax : ManagedMax;
        List<LoraSpec> specs = Adapters();
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var one = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        var group = new SimulatedDeviceGroup(2);
        using var sharded = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: group);
        Assert.Equal(2, sharded.Ranks.Count);
        Assert.IsType<ManagedGraphExecutor>(sharded.Ranks.Executor(1));
        string name = MiniMaxH3ImageFixture.Name(format);
        int compiles = 0, sums = 0;
        foreach (string geometry in new[] { "text", "edit" })
        {
            MiniMaxH3ImageGoldens.Request rq = Geometry(geometry);
            List<float[]> want = Steps(family, one, AdapterSet.Empty, rq);
            AssertClose(want, Steps(family, sharded, AdapterSet.Empty, rq, advanceWith: want), $"{name} {geometry} --tp 2", l2Bound, maxBound);
            compiles += 2;   // one program a rank, replayed by the second step
            sums += 2 * 2 * MiniMaxH3ImageFixture.Blocks;   // two steps, two all-reduces a block
            Assert.Equal(compiles, sharded.Compiles);
            Assert.Equal(sums, group.AllReduceCount);
        }

        using var oneSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), one.Config, BackendType.Cpu);
        using var shardedSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), sharded.Config, BackendType.Cpu);
        foreach (string geometry in new[] { "text", "edit" })
        {
            MiniMaxH3ImageGoldens.Request rq = Geometry(geometry);
            List<float[]> plain = Steps(family, one, AdapterSet.Empty, rq);
            List<float[]> want = Steps(family, one, oneSet.AdapterSet, rq);
            AssertClose(want, Steps(family, sharded, shardedSet.AdapterSet, rq, advanceWith: want), $"{name} {geometry} --tp 2 with LoRA",
                l2Bound, maxBound);
            // The adapters do move the velocity (the comparison is not of two unadapted steps).
            for (int i = 0; i < want.Count; i++) Assert.True(Compare(plain[i], want[i]).RelL2 > 1e-2);
        }
        Assert.Same(shardedSet, sharded.BoundLora);
        // A set change drops the plans: each rank compiled again for each geometry.
        Assert.Equal(compiles + 4, sharded.Compiles);

        // Without the adapters again: no rank keeps the set's plan or slices.
        List<float[]> after = Steps(family, one, AdapterSet.Empty, Geometry("text"));
        AssertClose(after, Steps(family, sharded, AdapterSet.Empty, Geometry("text"), advanceWith: after), $"{name} text --tp 2 after LoRA",
            l2Bound, maxBound);
        Assert.Null(sharded.BoundLora);
    }

    /// <summary>One sharded denoiser serves set A, then B, then A reloaded (new factors, possibly at the freed set's
    /// addresses): each request's velocities are those of a fresh sharded denoiser given that set alone, bit for bit, so no
    /// rank's plan or slice of an earlier set survives into a later request.</summary>
    [Fact]
    public void SwitchingAdaptersLeavesNoRankSliceBehind()
    {
        List<LoraSpec> specsA = Adapters(31), specsB = Adapters(41);
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        MiniMaxH3ImageGoldens.Request rq = Geometry("text");
        List<float[]> Fresh(IReadOnlyList<LoraSpec> specs)
        {
            using var d = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(2));
            using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), d.Config, BackendType.Cpu);
            return Steps(family, d, set.AdapterSet, rq);
        }
        List<float[]> freshA = Fresh(specsA), freshB = Fresh(specsB);

        using var served = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(2));
        var setA = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsA), served.Config, BackendType.Cpu);
        AssertBitExact(freshA, Steps(family, served, setA.AdapterSet, rq), "A");
        using (var setB = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsB), served.Config, BackendType.Cpu))
            AssertBitExact(freshB, Steps(family, served, setB.AdapterSet, rq), "A, then B");
        // The model's swap: release the binding, free the set, load the next.
        served.ReleaseLora();
        setA.Dispose();
        using var reloaded = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specsA), served.Config, BackendType.Cpu);
        AssertBitExact(freshA, Steps(family, served, reloaded.AdapterSet, rq), "A reloaded");
        Assert.Same(reloaded, served.BoundLora);
    }

    /// <summary>A rank's device holds the replicated weights (the token refiner, the patch and condition projections, the final
    /// layer, every block's norms and modulation) and half of every block projection: <see cref="IDenoiser.WeightBytes"/> is
    /// that share, the residency plan's budget for one device.</summary>
    [Fact]
    public void WeightBytesAreOneRanksShare()
    {
        using var fx = new MiniMaxH3ImageFixture(F32);
        using var one = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        using var sharded = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(2));
        const long H = MiniMaxH3ImageFixture.Hidden, I = MiniMaxH3ImageFixture.Inner, F = MiniMaxH3ImageFixture.Ffn;
        long projections = 4L * (H * 3 * I + I * H + H * 2 * F + F * H);   // a block's qkv, out, fc1 and fc2 in F32
        Assert.Equal(one.WeightBytes - MiniMaxH3ImageFixture.Blocks * projections / 2, sharded.WeightBytes);
        // The shards are copies of their runs: two ranks' copies of every block projection.
        Assert.Equal(sharded.DeviceCopyKeysForTest().Count,
            one.DeviceCopyKeysForTest().Count + 2 * 4 * MiniMaxH3ImageFixture.Blocks);
    }

    /// <summary>What cannot be sharded is refused when the denoiser is made: a group that does not divide the heads, a split
    /// that cuts the row-parallel products' quant blocks (the Q4_K fixture's 256 attention-output columns over two GPUs give
    /// each 128, half a block), a GGML backend (its ranked runner refuses the step's offload nodes), and a group with
    /// whole-block stages (the hybrid).</summary>
    [Fact]
    public void WhatCannotBeShardedIsRefused()
    {
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        Assert.Contains("divides its 4 attention heads", Assert.Throws<NotSupportedException>(() =>
            new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(3))).Message);
        using (var q4k = new MiniMaxH3ImageFixture(Q4_K))
        {
            string refusal = Assert.Throws<NotSupportedException>(() =>
                new MiniMaxH3ImageDenoiser(q4k.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(2))).Message;
            Assert.Contains("128 input columns of blocks.0.attn.out_proj.weight", refusal);
            Assert.Contains("256-element blocks", refusal);
        }
        Assert.Contains("ranked graph runner refuses programs with diffusion-transformer offload nodes", Assert.Throws<NotSupportedException>(() =>
            new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.GgmlCpu, tensorParallel: new SimulatedDeviceGroup(2))).Message);
        var devices = new IAllocator[] { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
        Assert.Contains("not both", Assert.Throws<NotSupportedException>(() =>
            new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, devices, tensorParallel: new SimulatedDeviceGroup(2))).Message);
        // A group of one is one device; the release's Q8_0 and every divisible split pass.
        using (var single = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu, tensorParallel: new SimulatedDeviceGroup(1)))
            Assert.Null(single.Ranks);
        using (var file = new TensorSharp.Models.MiniMaxH3.MiniMaxH3WeightFile(fx.DitPath))
        {
            Assert.Null(MiniMaxH3ImageDenoiser.TensorParallelRefusal(file, 2));
            Assert.Null(MiniMaxH3ImageDenoiser.TensorParallelRefusal(file, 4));
        }
    }

    /// <summary>The hybrid (--tp with --layer-split) is the contract's gap on direct CUDA: refused before anything loads.</summary>
    [Fact]
    public void TheHybridIsRefusedBeforeTheLoad()
    {
        using var dit = new MiniMaxH3ImageFixture(BF16, safetensors: true);
        var hybrid = Assert.Throws<NotSupportedException>(() => ModelBase.Create(dit.DitPath, BackendType.Cuda, tpDegree: 2, layerSplitDegree: 2));
        Assert.Contains("--tp 2 with --layer-split 2", hybrid.Message);
    }

    // ================================================================ the decoder on a tensor executor

    /// <summary>The decoder's program bound as a tensor executor binds it (its projections as weights, its F32 leaves as
    /// tensors on the device), here the managed executor: the frozen legacy pixels within the cross-lane bound, a program a
    /// row of tiles (the encoder on a tensor executor: MiniMaxH3ImageEditTests.VaeEncoderMatchesTheNumPyOracleOnTheManagedExecutor).</summary>
    [Theory]
    [InlineData("vae/canvas-32x40")]
    [InlineData("vae/split-4")]
    public void TheDecoderRunsOnATensorExecutor(string name)
    {
        MiniMaxH3ImageGoldens.VaeCase c = MiniMaxH3ImageGoldens.VaeCases.Single(v => v.Name == name);
        using var vae = new MiniMaxH3ImageVaeFixture(c.Blocks);
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        using var codec = new MiniMaxH3ImageVaeCodec(vae.VaePath, new ManagedGraphExecutor(cpu), cpu) { BatchLimit = c.Batch > 0 ? c.Batch : MiniMaxH3ImageVaeCodec.MaxBatch };
        float[] latent = MiniMaxH3ImageGoldens.NormalizedLatent(c.LatentH, c.LatentW, c.Seed);
        RgbImage got = codec.Decode(latent, c.LatentH, c.LatentW);
        MiniMaxH3ImageGoldens.Check(name, MiniMaxH3ImageGoldens.LaneKey(BackendType.Cpu), MiniMaxH3ImageGoldens.FileSha(vae.VaePath),
            new[] { got.Pixels }, 3e-3, output.WriteLine);
        Assert.True(codec.Compiles >= 1);
        // Released device copies are made again by the next decode, to the same pixels.
        codec.ReleaseDevice();
        Assert.Equal(got.Pixels, codec.Decode(latent, c.LatentH, c.LatentW).Pixels);
    }

    // ================================================================ direct CUDA (GPU validation pending)

    /// <summary>Direct CUDA against the managed executor (provisional bounds, to be measured on the GPU: the managed executor
    /// rounds a block format's activations and runs the attention's F16 ops, the CUDA one keeps F32 activations and serves the
    /// regions with its own kernels), with and without the adapters, wide and not.</summary>
    [CudaTheory]
    [InlineData(F32, false, 5e-3)]
    [InlineData(BF16, false, 5e-3)]
    [InlineData(BF16, true, 5e-3)]
    [InlineData(Q8_0, false, 3e-2)]
    [InlineData(Q4_K, true, 3e-2)]
    public void CudaStepsMatchTheManagedExecutor(int format, bool wide, double bound)
    {
        List<LoraSpec> specs = Adapters();
        using var allocator = new CudaAllocator(0);
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var managed = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        using var cuda = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, allocator: allocator);
        Assert.IsType<CudaGraphExecutor>(cuda.Executor);
        Assert.True(cuda.OnTensorExecutor);
        MiniMaxH3ImageGoldens.Request rq = Geometry(wide ? "wide" : "text");
        string what = $"cuda {MiniMaxH3ImageFixture.Name(format)}{(wide ? " wide" : "")}";
        List<float[]> want = Steps(family, managed, AdapterSet.Empty, rq);
        AssertClose(want, Steps(family, cuda, AdapterSet.Empty, rq, advanceWith: want), what, bound, bound);
        var executor = (CudaGraphExecutor)cuda.Executor;
        output.WriteLine($"{what}: {executor.DitRopesServedForTests} rotary attentions served");
        Assert.True(executor.DitRopesServedForTests > 0, "the blocks' attentions are the executor's own");
        Assert.Equal(1, cuda.Compiles);
        using var managedSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), managed.Config, BackendType.Cpu);
        using var cudaSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), cuda.Config, BackendType.Cuda);
        want = Steps(family, managed, managedSet.AdapterSet, rq);
        AssertClose(want, Steps(family, cuda, cudaSet.AdapterSet, rq, advanceWith: want), what + " with LoRA", bound, bound);
        Assert.Same(cudaSet, cuda.BoundLora);
        want = Steps(family, managed, AdapterSet.Empty, rq);
        AssertClose(want, Steps(family, cuda, AdapterSet.Empty, rq, advanceWith: want), what + " after LoRA", bound, bound);
    }

    /// <summary>On direct CUDA the weights' and the factors' device copies are in the resident-weight cache after a step, and
    /// gone after <see cref="MiniMaxH3ImageDenoiser.ReleaseDevice"/> (the next step makes them again, to the same bytes) and
    /// after the denoiser is disposed (a copy left keyed by a host address would serve the next file mapped there).</summary>
    [CudaFact]
    public void TheCudaDeviceCopiesGoWithTheDenoiser()
    {
        List<LoraSpec> specs = Adapters();
        using var allocator = new CudaAllocator(0);
        using var fx = new MiniMaxH3ImageFixture(Q4_K);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        var cuda = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, allocator: allocator);
        using var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), cuda.Config, BackendType.Cuda);
        int Resident(IReadOnlyList<IntPtr> keys) => keys.Count(k => CudaQuantizedOps.TryGetResidentDevicePtr(allocator, k, out _, out _));
        List<float[]> first = Steps(family, cuda, set.AdapterSet, Geometry("text"));
        IReadOnlyList<IntPtr> keys = cuda.DeviceCopyKeysForTest();
        Assert.True(Resident(keys) > 0, "a step leaves its weights' copies resident");
        cuda.ReleaseDevice();
        Assert.Equal(0, Resident(keys));
        AssertBitExact(first, Steps(family, cuda, set.AdapterSet, Geometry("text")), "after released device copies");
        keys = cuda.DeviceCopyKeysForTest();
        Assert.True(Resident(keys) > 0);
        cuda.Dispose();
        Assert.Equal(0, Resident(keys));
    }

    /// <summary>The blocks over two CUDA devices as whole-block stages (each stage's executor the one a device runs alone):
    /// every step equals one device's within 1e-5 (provisional), with and without the adapters.</summary>
    [MultiCudaTheory(2)]
    [InlineData(false)]
    [InlineData(true)]
    public void WholeBlockStagesOnTwoCudaDevicesMatchOneDevice(bool lora)
    {
        List<LoraSpec> specs = Adapters();
        using var first = new CudaAllocator(0);
        using var second = new CudaAllocator(1);
        using var fx = new MiniMaxH3ImageFixture(Q8_0);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        List<float[]> want;
        using (var one = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, allocator: first))
        using (var set = lora ? MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), one.Config, BackendType.Cuda) : null)
            want = Steps(family, one, set?.AdapterSet ?? AdapterSet.Empty, Geometry("text"));
        using var staged = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, devices: new IAllocator[] { first, second });
        Assert.Equal(2, staged.Stages.Count);
        Assert.IsType<CudaGraphExecutor>(staged.Stages.Executor(1));
        using var stagedSet = lora ? MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), staged.Config, BackendType.Cuda) : null;
        AssertClose(want, Steps(family, staged, stagedSet?.AdapterSet ?? AdapterSet.Empty, Geometry("text"), advanceWith: want),
            $"cuda stages{(lora ? " with LoRA" : "")}", 1e-5, 1e-5);
        Assert.Equal(2, staged.Compiles);
    }

    /// <summary>The denoiser sharded over direct CUDA's group of two GPUs (each rank on its own CUDA executor, the partials
    /// summed by the group's device-ordered peer reduction) against one GPU: every step within 1e-4 (provisional: the two
    /// partials' F32 sums against one product's), with the adapters and after them, plain and wide (the served regions).</summary>
    [MultiCudaTheory(2)]
    [InlineData(Q8_0, false)]
    [InlineData(BF16, true)]
    public void ShardedStepsOnTwoCudaDevicesMatchOneDevice(int format, bool wide)
    {
        List<LoraSpec> specs = Adapters();
        using var group = new TensorSharp.Cuda.TensorParallelGroup(2);
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        MiniMaxH3ImageGoldens.Request rq = Geometry(wide ? "wide" : "text");
        string what = $"cuda --tp 2 {MiniMaxH3ImageFixture.Name(format)}{(wide ? " wide" : "")}";
        List<float[]> want, wantLora;
        using (var one = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, allocator: group.GetAllocator(0)))
        using (var set = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), one.Config, BackendType.Cuda))
        {
            want = Steps(family, one, AdapterSet.Empty, rq);
            wantLora = Steps(family, one, set.AdapterSet, rq);
        }
        using var sharded = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cuda, tensorParallel: group);
        Assert.Equal(2, sharded.Ranks.Count);
        AssertClose(want, Steps(family, sharded, AdapterSet.Empty, rq, advanceWith: want), what, 1e-4, 1e-4);
        Assert.Equal(2, sharded.Compiles);
        using var shardedSet = MiniMaxH3ImageLoraSet.Load(LoraCliFlags.Resolve(specs), sharded.Config, BackendType.Cuda);
        AssertClose(wantLora, Steps(family, sharded, shardedSet.AdapterSet, rq, advanceWith: wantLora), what + " with LoRA", 1e-4, 1e-4);
        AssertClose(want, Steps(family, sharded, AdapterSet.Empty, rq, advanceWith: want), what + " after LoRA", 1e-4, 1e-4);
    }

    /// <summary>The decoder on direct CUDA against the frozen legacy pixels (provisional cross-lane bound), and its device
    /// copies released and made again to the same pixels.</summary>
    [CudaFact]
    public void TheDecoderOnCudaIsTheLegacyDecoder()
    {
        MiniMaxH3ImageGoldens.VaeCase c = MiniMaxH3ImageGoldens.VaeCases.Single(v => v.Name == "vae/canvas-32x40");
        using var allocator = new CudaAllocator(0);
        using var vae = new MiniMaxH3ImageVaeFixture(c.Blocks);
        using var codec = new MiniMaxH3ImageVaeCodec(vae.VaePath, BackendType.Cuda, allocator);
        float[] latent = MiniMaxH3ImageGoldens.NormalizedLatent(c.LatentH, c.LatentW, c.Seed);
        RgbImage got = codec.Decode(latent, c.LatentH, c.LatentW);
        MiniMaxH3ImageGoldens.Check(c.Name, MiniMaxH3ImageGoldens.LaneKey(BackendType.Cuda), MiniMaxH3ImageGoldens.FileSha(vae.VaePath),
            new[] { got.Pixels }, 3e-3, output.WriteLine);
        codec.ReleaseDevice();
        Assert.Equal(got.Pixels, codec.Decode(latent, c.LatentH, c.LatentW).Pixels);
    }

    /// <summary>The model on direct CUDA with --tp 2: it loads with the group in the pipeline's load (nothing on a device before
    /// the first image), it takes edits (the VAE encoder runs on the tensor executor), and a Q4_K denoiser whose attention output a 2-way split
    /// would cut is refused at load.</summary>
    [MultiCudaFact(2)]
    public void TheModelShardsItsDenoiserOverTwoCudaGpus()
    {
        using var dit = new MiniMaxH3ImageFixture(Q8_0);
        using var vae = new MiniMaxH3ImageVaeFixture(2);
        System.IO.File.WriteAllBytes(System.IO.Path.Combine(dit.Dir, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"), new byte[16]);
        System.IO.File.Copy(vae.VaePath, System.IO.Path.Combine(dit.Dir, "minimax_h3_image_vae_fp16.safetensors"));
        using (ModelBase model = ModelBase.Create(dit.DitPath, BackendType.Cuda, tpDegree: 2))
        {
            var runtime = Assert.IsType<MiniMaxH3ImageDiffusionModel>(model);
            Assert.True(runtime.SupportsImageEdit);
            Assert.False(runtime.Pipeline.IsResident(DiffusionComponent.Denoiser));
        }
        using var q4k = new MiniMaxH3ImageFixture(Q4_K);
        System.IO.File.WriteAllBytes(System.IO.Path.Combine(q4k.Dir, "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"), new byte[16]);
        System.IO.File.Copy(vae.VaePath, System.IO.Path.Combine(q4k.Dir, "minimax_h3_image_vae_fp16.safetensors"));
        Assert.Contains("256-element blocks", Assert.Throws<ModelLoadRefusedException>(() =>
            ModelBase.Create(q4k.DitPath, BackendType.Cuda, tpDegree: 2)).Message);
    }
}
