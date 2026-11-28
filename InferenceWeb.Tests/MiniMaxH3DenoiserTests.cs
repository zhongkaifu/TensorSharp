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
using System.Threading;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3's audio-video denoiser on the diffusion runtime (MiniMaxH3Denoiser: one GraphProgram a step) on the synthetic
/// fixture (<see cref="MiniMaxH3VideoFixture"/>): its steps against the retired DiT's, frozen before it was deleted
/// (<see cref="MiniMaxH3VideoGoldens"/>: the native whole graph on the GGML lanes, the direct DiT on cpu and cuda), byte for
/// byte on a lane where the runtime gave the legacy bytes when they were frozen and within a bound elsewhere; and its plans,
/// LoRA binding, streams and refusals.
/// </summary>
public sealed class MiniMaxH3DenoiserTests(ITestOutputHelper output)
{
    /// <summary>Off the lane a case was frozen on (another machine, or the tensor executors against the retired direct DiT,
    /// whose kernels sum in other orders), the bound on its largest sampled difference over the largest legacy value.</summary>
    private static double Bound(MiniMaxH3VideoGoldens.StepCase c, BackendType backend) =>
        (c.Projections is MiniMaxH3ImageFixture.Q8_0 or MiniMaxH3ImageFixture.Q4_K ? 1e-2 : 3e-3) * (c.Lora ? 2 : 1);

    public static TheoryData<string> CaseNames => new(MiniMaxH3VideoGoldens.Cases.Select(c => c.Name));

    private static (float[] Video, float[] Audio) Run(MiniMaxH3Denoiser dit, MiniMaxH3VideoGoldens.StepCase c, MiniMaxH3LoraSet lora,
        float[] text = null)
    {
        var (v, a) = dit.Forward(lora, c.Video, c.Shape.VideoTokenCount, c.Audio, c.Shape.AudioTokenCount, text ?? c.Text,
            MiniMaxH3VideoGoldens.StepCase.TextTokens, c.Layout(), c.Sigma, c.Condition, c.ConditionCount, c.ConditionAudio,
            c.ConditionAudioCount, c.Chunks);
        return ((float[])v.Clone(), a);
    }

    private void CheckCase(string name, BackendType backend, IAllocator allocator = null)
    {
        MiniMaxH3VideoGoldens.StepCase c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == name);
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        string manifest = c.Lora ? fx.WriteLora() : null;
        string weights = c.Lora ? Path.Combine(fx.Dir, MiniMaxH3LoraFixtures.Weights8) : null;
        string key = MiniMaxH3VideoGoldens.FixtureKey(fx.DitPath, weights);
        string lane = MiniMaxH3VideoGoldens.LaneKey(backend);
        MiniMaxH3VideoGoldens.Case frozen = MiniMaxH3VideoGoldens.Find(name, lane, key);
        Assert.True(frozen != null, $"no frozen legacy step for {name} from these fixture bytes (lanes: {string.Join("; ", MiniMaxH3VideoGoldens.Lanes)})");
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, backend, allocator);
        using MiniMaxH3LoraSet lora = c.Lora
            ? MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config, MiniMaxH3Partition.FirstLastFrame,
                backend, tensorExecutor: dit.OnTensorExecutor)
            : null;
        var (video, audio) = Run(dit, c, lora);
        bool exact = frozen.Lane == lane && MiniMaxH3ImageGoldens.Sha(video) == frozen.Outputs[0].Sha256
                     && MiniMaxH3ImageGoldens.Sha(audio) == frozen.Outputs[1].Sha256;
        output.WriteLine($"[h3-denoiser] {name} on {lane} against {frozen.Lane}: runtime bytes {(exact ? "equal" : "differ from")} the legacy ones");
        // Where every lane's legacy step lies from this one (the retired backends disagreed among themselves too).
        foreach (MiniMaxH3VideoGoldens.Case other in MiniMaxH3VideoGoldens.Everywhere(name, key))
            output.WriteLine($"[h3-denoiser] {name} on {lane} vs legacy {other.Lane.Split('|')[0]}: video " +
                $"{MiniMaxH3VideoGoldens.SampledRelative(other.Outputs[0], video):E2}, audio {MiniMaxH3VideoGoldens.SampledRelative(other.Outputs[1], audio):E2}");
        var asImage = new MiniMaxH3ImageGoldens.Case(frozen.Lane, frozen.FixtureSha256, frozen.RuntimeBitExact, frozen.Outputs);
        // The tensor executors take the retired direct DiT's arithmetic on their backend (its products and its attention):
        // against the lane the retired direct DiT was frozen on, on this machine, their steps are held to 1e-5 (measured
        // within 5.3e-7 on the A40 and the Xeon), except where the managed executor's block-format products quantize their
        // activations: one rounding flipped by a last-bit difference upstream moves the step by up to 7.6e-4 (the Xeon's
        // Q8_0 case; its three Q4_K cases are within 3.7e-7), so 5e-3 there. Elsewhere the cross-backend bound.
        bool tensor = backend is BackendType.Cuda or BackendType.Cpu;
        bool roundsActivations = backend == BackendType.Cpu && c.Projections is MiniMaxH3ImageFixture.Q8_0 or MiniMaxH3ImageFixture.Q4_K;
        double bound = tensor && frozen.Lane == lane ? (roundsActivations ? 5e-3 : 1e-5) : Bound(c, backend);
        MiniMaxH3ImageGoldens.CheckAgainst(asImage, name, lane, key, new[] { video, audio }, bound, output.WriteLine);
    }

    // ================================================================ against the frozen legacy steps

    /// <summary>The GGML graph executor on the process's pinned backend (TS_TEST_GGML_BACKEND; GgmlCpu by default) against the
    /// retired native whole graph's steps.</summary>
    [Theory]
    [MemberData(nameof(CaseNames))]
    public void GraphExecutorStepsMatchTheFrozenLegacySteps(string name) => CheckCase(name, TestGates.PinnedGgmlBackend);

    /// <summary>The managed tensor executor (the pure-C# cpu backend, its own products) against the retired direct CPU DiT's steps.</summary>
    [Theory]
    [MemberData(nameof(CaseNames))]
    public void ManagedExecutorStepsMatchTheFrozenLegacySteps(string name) => CheckCase(name, BackendType.Cpu);

    /// <summary>The direct CUDA tensor executor (bounded pedantic F32 products) against the retired direct CUDA DiT's steps,
    /// every case on one allocator: each fixture is a new file, often mapped where the last one was, so a device copy a
    /// disposed denoiser left in the resident-weight cache (keyed by host address) would be bound in its place.</summary>
    [CudaFact]
    public void CudaExecutorStepsMatchTheFrozenLegacySteps()
    {
        using var allocator = new CudaAllocator(0);
        foreach (var c in MiniMaxH3VideoGoldens.Cases) CheckCase(c.Name, BackendType.Cuda, allocator);
    }

    /// <summary>The tensor executors serve a LoraLinear region in place (the product, then each part's update added into its
    /// slot's columns): the bytes of the region's ops, which build each slot's sum and concatenate them, for q, k and v as
    /// three slots and the single-slot projections, on every LoRA case.</summary>
    private static void CheckServedLoraLinear(BackendType backend, IAllocator allocator)
    {
        foreach (MiniMaxH3VideoGoldens.StepCase c in MiniMaxH3VideoGoldens.Cases.Where(x => x.Lora))
        {
            using var fx = new MiniMaxH3VideoFixture(c.Projections);
            string manifest = fx.WriteLora();
            (float[] Video, float[] Audio) Step(bool served)
            {
                using var dit = new MiniMaxH3Denoiser(fx.DitPath, backend, allocator);
                var executor = (TensorSharp.Graph.TensorGraphExecutor)dit.Executor;
                executor.LoraLinearInPlaceDisabledForTests = !served;
                using MiniMaxH3LoraSet lora = MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config,
                    MiniMaxH3Partition.FirstLastFrame, backend, tensorExecutor: dit.OnTensorExecutor);
                long before = executor.LoraLinearsServedForTests;
                var result = Run(dit, c, lora);
                // Every adapted projection of the two blocks and the refiner block: four regions a block.
                Assert.Equal(served ? 4L * (MiniMaxH3VideoFixture.Blocks + MiniMaxH3VideoFixture.RefinerBlocks) : 0L,
                    executor.LoraLinearsServedForTests - before);
                executor.LoraLinearInPlaceDisabledForTests = false;
                return result;
            }
            var ops = Step(served: false);
            var served = Step(served: true);
            TinyH3Step.AssertBits(ops.Video, served.Video, c.Name + " video");
            TinyH3Step.AssertBits(ops.Audio, served.Audio, c.Name + " audio");
        }
    }

    [Fact]
    public void TheManagedExecutorsServedLoraLinearGivesItsOpsBytes() => CheckServedLoraLinear(BackendType.Cpu, null);

    [CudaFact]
    public void TheCudaExecutorsServedLoraLinearGivesItsOpsBytes()
    {
        using var allocator = new CudaAllocator(0);
        CheckServedLoraLinear(BackendType.Cuda, allocator);
    }

    /// <summary>The tensor executors serve the trunk's and the refiner's attention in F32 (on the CPU the retired direct DiT's
    /// blocked attention; on CUDA batched GEMMs over groups of heads in chunks of query rows, each block's with its q/k norm
    /// and rotation on the retired DiT's kernels): within the region ops' F16 K/V
    /// rounding of them, on every case (5e-3 of the largest value; 2e-2 where the managed executor's Q8_0/Q4_K products
    /// quantize the activations that difference reaches, measured within 7.2e-3).</summary>
    private void CheckServedAttention(BackendType backend, Func<IAllocator> newAllocator)
    {
        foreach (MiniMaxH3VideoGoldens.StepCase c in MiniMaxH3VideoGoldens.Cases)
        {
            using var fx = new MiniMaxH3VideoFixture(c.Projections);
            string manifest = c.Lora ? fx.WriteLora() : null;
            (float[] Video, float[] Audio) Step(bool served)
            {
                // An executor of its own a step: one that ran the same program before replays the plan it recorded then.
                IAllocator allocator = newAllocator?.Invoke();
                try
                {
                    using var dit = new MiniMaxH3Denoiser(fx.DitPath, backend, allocator);
                    var executor = (TensorSharp.Graph.TensorGraphExecutor)dit.Executor;
                    executor.DitAttentionServedDisabledForTests = !served;
                    using MiniMaxH3LoraSet lora = c.Lora
                        ? MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config,
                            MiniMaxH3Partition.FirstLastFrame, backend, tensorExecutor: dit.OnTensorExecutor)
                        : null;
                    long before = executor.DitAttentionsServedForTests;
                    long ropesBefore = (executor as CudaGraphExecutor)?.DitRopesServedForTests ?? 0;
                    var result = Run(dit, c, lora);
                    // One attention a block and a refiner block; on CUDA each block's with its rotary prep.
                    Assert.Equal(served ? (long)(MiniMaxH3VideoFixture.Blocks + MiniMaxH3VideoFixture.RefinerBlocks) : 0L,
                        executor.DitAttentionsServedForTests - before);
                    if (executor is CudaGraphExecutor cuda)
                        Assert.Equal(served ? (long)MiniMaxH3VideoFixture.Blocks : 0L, cuda.DitRopesServedForTests - ropesBefore);
                    executor.DitAttentionServedDisabledForTests = false;
                    return result;
                }
                finally
                {
                    (allocator as IDisposable)?.Dispose();
                }
            }
            var ops = Step(served: false);
            var served = Step(served: true);
            foreach (var (got, want, what) in new[] { (served.Video, ops.Video, "video"), (served.Audio, ops.Audio, "audio") })
            {
                double err = 0, peak = 0;
                for (int i = 0; i < want.Length; i++) { err = Math.Max(err, Math.Abs(got[i] - want[i])); peak = Math.Max(peak, Math.Abs(want[i])); }
                output.WriteLine($"[h3-attention] {backend} {c.Name} {what}: {err / peak:E2} of the largest value from the region's ops");
                double bound = backend == BackendType.Cpu && c.Projections is MiniMaxH3ImageFixture.Q8_0 or MiniMaxH3ImageFixture.Q4_K ? 2e-2 : 5e-3;
                Assert.True(err / peak < bound, $"{c.Name} {what}: {err / peak:E3}");
                Assert.True(err > 0, $"{c.Name} {what}: the served attention gave the region ops' bytes (F32 against their F16 K/V)");
            }
        }
    }

    [Fact]
    public void TheManagedExecutorsServedAttentionMatchesItsOps() => CheckServedAttention(BackendType.Cpu, null);

    [CudaFact]
    public void TheCudaExecutorsServedAttentionMatchesItsOps() => CheckServedAttention(BackendType.Cuda, () => new CudaAllocator(0));

    /// <summary>The tensor executors serve the blocks' adaLN modulations and gated residuals as the retired direct DiT ran them
    /// (the RMS norm with its weight in one pass, then each segment's rows modulated or gated in place): every one of the four
    /// regions a block, within the rounding of the region's ops (n + n scale + shift, the norm and its weight as two ops) on
    /// every case.</summary>
    private void CheckServedModulation(BackendType backend, Func<IAllocator> newAllocator)
    {
        foreach (MiniMaxH3VideoGoldens.StepCase c in MiniMaxH3VideoGoldens.Cases)
        {
            using var fx = new MiniMaxH3VideoFixture(c.Projections);
            string manifest = c.Lora ? fx.WriteLora() : null;
            (float[] Video, float[] Audio) Step(bool served)
            {
                IAllocator allocator = newAllocator?.Invoke();
                try
                {
                    using var dit = new MiniMaxH3Denoiser(fx.DitPath, backend, allocator);
                    var executor = (TensorSharp.Graph.TensorGraphExecutor)dit.Executor;
                    executor.DitModulationServedDisabledForTests = !served;
                    using MiniMaxH3LoraSet lora = c.Lora
                        ? MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config,
                            MiniMaxH3Partition.FirstLastFrame, backend, tensorExecutor: dit.OnTensorExecutor)
                        : null;
                    long before = executor.DitModulationsServedForTests;
                    var result = Run(dit, c, lora);
                    Assert.Equal(served ? 4L * MiniMaxH3VideoFixture.Blocks : 0L, executor.DitModulationsServedForTests - before);
                    executor.DitModulationServedDisabledForTests = false;
                    return result;
                }
                finally
                {
                    (allocator as IDisposable)?.Dispose();
                }
            }
            var ops = Step(served: false);
            var served = Step(served: true);
            foreach (var (got, want, what) in new[] { (served.Video, ops.Video, "video"), (served.Audio, ops.Audio, "audio") })
            {
                double err = 0, peak = 0;
                for (int i = 0; i < want.Length; i++) { err = Math.Max(err, Math.Abs(got[i] - want[i])); peak = Math.Max(peak, Math.Abs(want[i])); }
                output.WriteLine($"[h3-modulation] {backend} {c.Name} {what}: {err / peak:E2} of the largest value from the regions' ops");
                double bound = backend == BackendType.Cpu && c.Projections is MiniMaxH3ImageFixture.Q8_0 or MiniMaxH3ImageFixture.Q4_K ? 2e-2 : 1e-4;
                Assert.True(err / peak < bound, $"{c.Name} {what}: {err / peak:E3}");
            }
        }
    }

    [Fact]
    public void TheManagedExecutorsServedModulationMatchesItsOps() => CheckServedModulation(BackendType.Cpu, null);

    [CudaFact]
    public void TheCudaExecutorsServedModulationMatchesItsOps() => CheckServedModulation(BackendType.Cuda, () => new CudaAllocator(0));

    /// <summary>On the CUDA tensor executor the weights' and the LoRA factors' device copies are in the resident-weight
    /// cache after a step, and gone after <see cref="MiniMaxH3Denoiser.ReleaseDevice"/> and after the denoiser is disposed
    /// (they had stayed: the device memory stayed taken, and a 0.64-relative-error step bound another fixture's copies).</summary>
    [CudaFact]
    public void TheCudaExecutorsDeviceCopiesGoWithTheDenoiser()
    {
        MiniMaxH3VideoGoldens.StepCase c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "lora-t2v-q4_k-s1");
        using var allocator = new CudaAllocator(0);
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        string manifest = fx.WriteLora();
        var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, allocator);
        using MiniMaxH3LoraSet lora = MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config,
            MiniMaxH3Partition.FirstLastFrame, BackendType.Cuda, tensorExecutor: dit.OnTensorExecutor);
        int Resident(IReadOnlyList<IntPtr> keys) => keys.Count(k => CudaQuantizedOps.TryGetResidentDevicePtr(allocator, k, out _, out _));
        var (first, _) = Run(dit, c, lora);
        IReadOnlyList<IntPtr> keys = dit.DeviceCopyKeysForTest();
        Assert.True(Resident(keys) > 0, "a step leaves its weights' copies resident");
        dit.ReleaseDevice();
        Assert.Equal(0, Resident(keys));
        TinyH3Step.AssertBits(first, Run(dit, c, lora).Video, "after released device copies");
        Assert.True(Resident(dit.DeviceCopyKeysForTest()) > 0);
        keys = dit.DeviceCopyKeysForTest();
        dit.Dispose();
        Assert.Equal(0, Resident(keys));
    }

    // ================================================================ plans, LoRA binding, streams, refusals

    /// <summary>A geometry's steps share one plan (a sigma changes only the inputs), another text length compiles a new one, a
    /// released workspace compiles again with the same results, as do released device copies.</summary>
    [Fact]
    public void AGeometrysStepsShareOnePlan()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "keyframes-bf16-s075");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        var first = Run(dit, c, null);
        var other = Run(dit, c with { Sigma = .5f }, null);
        Assert.Equal(1, dit.Compiles);
        Assert.NotEqual(first.Video, other.Video);
        var again = Run(dit, c, null);
        Assert.Equal(1, dit.Compiles);
        TinyH3Step.AssertBits(first.Video, again.Video, "replayed video");
        TinyH3Step.AssertBits(first.Audio, again.Audio, "replayed audio");

        dit.ReleaseWorkspace();
        TinyH3Step.AssertBits(first.Video, Run(dit, c, null).Video, "after a released workspace");
        Assert.Equal(2, dit.Compiles);
        dit.ReleaseDevice();
        TinyH3Step.AssertBits(first.Video, Run(dit, c, null).Video, "after released device copies");
        Assert.Equal(3, dit.Compiles);
    }

    /// <summary>The token refiner (its own program) runs once a prompt and LoRA set: the clip's later steps, and a clip after a
    /// released workspace, take its rows as they are (the retired direct DiT's text cache); other text rows run it again, and
    /// the first text's rows give the first step's bytes again.</summary>
    [Fact]
    public void TheTokenRefinerRunsOnceAPrompt()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "keyframes-bf16-s075");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        var first = Run(dit, c, null);
        Run(dit, c with { Sigma = .5f }, null);
        Assert.Equal(1, dit.Refinements);
        dit.ReleaseWorkspace();
        TinyH3Step.AssertBits(first.Video, Run(dit, c, null).Video, "after a released workspace");
        Assert.Equal(1, dit.Refinements);
        float[] other = c.Text;
        other[0] += 1f;
        var changed = Run(dit, c, null, other);
        Assert.Equal(2, dit.Refinements);
        Assert.NotEqual(first.Video, changed.Video);
        TinyH3Step.AssertBits(first.Video, Run(dit, c, null).Video, "the first text again");
        Assert.Equal(3, dit.Refinements);
    }

    /// <summary>The audio head's scale is applied after the program, so every sigma replays one plan; it is the retired
    /// forward's -dsigma_audio/dsigma_video at the step's sigma.</summary>
    [Fact]
    public void TheAudioPredictionIsScaledByTheShiftSlopeOutsideTheProgram()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "t2v-bf16-s1");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
        float[] atOne = Run(dit, c, null).Audio;
        // The same packed layout (its timesteps are sigma 1's) at another sigma: only the scale differs.
        var (_, at06) = dit.Forward(null, c.Video, c.Shape.VideoTokenCount, c.Audio, c.Shape.AudioTokenCount, c.Text,
            MiniMaxH3VideoGoldens.StepCase.TextTokens, c.Layout(), .6f);
        Assert.Equal(1, dit.Compiles);
        float s1 = -MiniMaxH3Scheduler.ShiftSlope(1f), s06 = -MiniMaxH3Scheduler.ShiftSlope(.6f);
        for (int i = 0; i < atOne.Length; i++)
            Assert.Equal(atOne[i] / s1 * s06, at06[i], 1e-4f * Math.Max(1f, Math.Abs(at06[i])));
    }

    /// <summary>Step reads its audio from the streams and returns its prediction as the audio stream; it refuses a layout
    /// another family packed and a step without its audio.</summary>
    [Fact]
    public void StepReadsAndReturnsTheAudioStream()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "ref2va-bf16-s075");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
        var expected = Run(dit, c, null);
        var text = new ConditioningSet(c.Text, MiniMaxH3VideoGoldens.StepCase.TextTokens, MiniMaxH3VideoFixture.TextDim, c.TextModalities);
        MiniMaxH3PackedLayout packed = c.Layout();
        var layout = new DiffusionLayout
        {
            Rows = packed.TokenCount, TargetStart = packed.VideoSpan.Start, TargetRows = packed.VideoSpan.Length,
            FamilyLayout = new MiniMaxH3StepLayout
            {
                Layout = packed, ConditionTokens = c.Condition, ConditionCount = c.ConditionCount, ConditionAudio = c.ConditionAudio,
                ConditionAudioCount = c.ConditionAudioCount, Chunks = c.Chunks,
            },
        };
        var recipe = new SamplingRecipe { Schedule = DiffusionSchedule.Explicit, Steps = 1, Sigmas = new[] { c.Sigma, 0f } };
        var inputs = new DenoiseStepInputs
        {
            Latent = c.Video, Conditioning = text, Layout = layout, Step = new StepState(0, 1, c.Sigma, 0f, 0, 0), Recipe = recipe,
            Streams = new Dictionary<string, float[]> { [MiniMaxH3Denoiser.AudioStream] = c.Audio },
        };
        StepOutput step = dit.Step(inputs, AdapterSet.Empty, CancellationToken.None);
        TinyH3Step.AssertBits(expected.Video, step.Velocity, "video");
        TinyH3Step.AssertBits(expected.Audio, step.Streams[MiniMaxH3Denoiser.AudioStream], "audio");
        Assert.Equal(new[] { MiniMaxH3Denoiser.AudioStream }, dit.Caps.Streams);

        Assert.Throws<ArgumentException>(() => dit.Step(new DenoiseStepInputs
        {
            Latent = c.Video, Conditioning = text, Layout = layout, Step = inputs.Step, Recipe = recipe,
        }, AdapterSet.Empty, CancellationToken.None));
        Assert.Throws<ArgumentException>(() => dit.Step(new DenoiseStepInputs
        {
            Latent = c.Video, Conditioning = text, Layout = new DiffusionLayout { Rows = 1, TargetStart = 0, TargetRows = 1 },
            Step = inputs.Step, Recipe = recipe, Streams = inputs.Streams,
        }, AdapterSet.Empty, CancellationToken.None));
    }

    /// <summary>A conditioning chunk list that does not cover the conditioning rows is refused before anything runs.</summary>
    [Fact]
    public void ChunksThatDoNotCoverTheConditioningAreRefused()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "ref2va-bf16-s075");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
        var e = Assert.Throws<ArgumentException>(() => dit.Forward(null, c.Video, c.Shape.VideoTokenCount, c.Audio, c.Shape.AudioTokenCount,
            c.Text, MiniMaxH3VideoGoldens.StepCase.TextTokens, c.Layout(), c.Sigma, c.Condition, c.ConditionCount, c.ConditionAudio,
            c.ConditionAudioCount, new[] { (1, 4), (0, 3) }));
        Assert.Contains("do not cover", e.Message);
        Assert.Throws<ArgumentException>(() => dit.Forward(null, c.Video, c.Shape.VideoTokenCount, c.Audio, c.Shape.AudioTokenCount,
            c.Text, MiniMaxH3VideoGoldens.StepCase.TextTokens, c.Layout(), c.Sigma, c.Condition, c.ConditionCount, c.ConditionAudio,
            c.ConditionAudioCount));
        Assert.Equal(0, dit.Compiles);
    }

    /// <summary>The block limit of a model slice's oracle runs the first blocks only, as a program of its own.</summary>
    [Fact]
    public void ABlockLimitRunsTheFirstBlocks()
    {
        var c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == "t2v-bf16-s060");
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
        var all = Run(dit, c, null);
        var (one, _) = dit.Forward(c.Video, c.Shape.VideoTokenCount, c.Audio, c.Shape.AudioTokenCount, c.Text,
            MiniMaxH3VideoGoldens.StepCase.TextTokens, c.Layout(), c.Sigma, blockLimit: 1);
        Assert.Equal(2, dit.Compiles);
        Assert.NotEqual(all.Video, one);
    }

    // ================================================================ whole-block stages (--layer-split N)

    /// <summary>A case's steps at its sigma and at half of it, as a sampler's: the second replays the plan its geometry
    /// compiled (at sigma 1 a layout packs its timesteps differently: another geometry, as a clip's first step is).</summary>
    private static List<(float[] Video, float[] Audio)> Steps(MiniMaxH3Denoiser dit, MiniMaxH3VideoGoldens.StepCase c, MiniMaxH3LoraSet lora) =>
        new() { Run(dit, c, lora), Run(dit, c with { Sigma = c.Sigma * .5f }, lora) };

    private static MiniMaxH3LoraSet LoadLora(string manifest, MiniMaxH3Denoiser dit, BackendType backend) =>
        MiniMaxH3LoraSet.Load(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }), dit.Config, MiniMaxH3Partition.FirstLastFrame,
            backend, tensorExecutor: dit.OnTensorExecutor);

    private static void AssertStepBits(List<(float[] Video, float[] Audio)> want, List<(float[] Video, float[] Audio)> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            TinyH3Step.AssertBits(want[i].Video, got[i].Video, $"{what}, step {i + 1} video");
            TinyH3Step.AssertBits(want[i].Audio, got[i].Audio, $"{what}, step {i + 1} audio");
        }
    }

    /// <summary>The largest difference over the largest value, and the L2 norm of the difference over the reference's.</summary>
    private static (double Rel, double RelL2) Compare(float[] want, float[] got)
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

    /// <summary>Each step's video and audio against one device's, within <paramref name="bound"/> of the largest value; the
    /// measured errors logged.</summary>
    private void AssertStepsClose(List<(float[] Video, float[] Audio)> want, List<(float[] Video, float[] Audio)> got, string what,
        double bound = 1e-5)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            var (video, videoL2) = Compare(want[i].Video, got[i].Video);
            var (audio, audioL2) = Compare(want[i].Audio, got[i].Audio);
            output.WriteLine($"[h3-stages] {what}, step {i + 1}: stages vs one device: video max {video:E3} (L2 {videoL2:E3}), " +
                             $"audio max {audio:E3} (L2 {audioL2:E3})");
            Assert.True(video <= bound && audio <= bound, $"{what}, step {i + 1}: video {video:E3}, audio {audio:E3}");
        }
    }

    /// <summary>
    /// The trunk's blocks over two devices as whole-block stages (DenoiserStages: one program a stage, the packed sequence's
    /// residual stream handed on as F32 rows; the token refiner and the patch projections with the first stage, the final
    /// layer and both heads with the last), on the managed executor of two host "devices": every case's every step, video
    /// and audio, equals one device's bit for bit, with and without the LoRA plug-in (each stage binds its own blocks'
    /// factors, the refiner's with the first) and again after it; each stage compiles where one device compiles (once a
    /// geometry and LoRA set), and the refiner runs once a prompt; released device copies give the same bytes. The stages'
    /// weights are the denoiser's, each leaf on one stage.
    /// </summary>
    [Fact]
    public void WholeBlockStagesMatchOneDeviceOnTheManagedExecutor()
    {
        foreach (MiniMaxH3VideoGoldens.StepCase c in MiniMaxH3VideoGoldens.Cases)
        {
            using var fx = new MiniMaxH3VideoFixture(c.Projections);
            using var one = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
            var devices = new IAllocator[] { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
            using var staged = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, devices: devices);
            Assert.Equal(2, staged.Stages.Count);
            Assert.Equal((0, 1), (staged.Stages.First(0), staged.Stages.End(0)));
            Assert.Equal((1, 2), (staged.Stages.First(1), staged.Stages.End(1)));
            Assert.True(staged.OnTensorExecutor);
            Assert.True(staged.WeightBytes < one.WeightBytes, "a stage holds its share of the weights, not all of them");

            List<(float[] Video, float[] Audio)> want = Steps(one, c, null);
            AssertStepBits(want, Steps(staged, c, null), c.Name + ", stages");
            // One program a stage wherever one device compiles one (a geometry), replayed by the steps that follow.
            Assert.InRange(one.Compiles, 1, 2);
            Assert.Equal(2 * one.Compiles, staged.Compiles);
            Assert.Equal(1, one.Refinements);
            Assert.Equal(one.Refinements, staged.Refinements);

            if (c.Lora)
            {
                string manifest = fx.WriteLora();
                using MiniMaxH3LoraSet oneSet = LoadLora(manifest, one, BackendType.Cpu);
                using MiniMaxH3LoraSet stagedSet = LoadLora(manifest, staged, BackendType.Cpu);
                AssertStepBits(Steps(one, c, oneSet), Steps(staged, c, stagedSet), c.Name + ", stages with the LoRA plug-in");
                Assert.Same(stagedSet, staged.BoundLora);
                Assert.Equal(2 * one.Compiles, staged.Compiles);
                Assert.Equal(one.Refinements, staged.Refinements);
                // Without the adapters again: no stage keeps the set's plan or factors.
                AssertStepBits(Steps(one, c, null), Steps(staged, c, null), c.Name + ", stages after the LoRA plug-in");
                Assert.Null(staged.BoundLora);
                Assert.Equal(2 * one.Compiles, staged.Compiles);
            }

            // Released device copies (every stage's plan with them): the next steps compile each stage again, to the same bytes.
            one.ReleaseDevice();
            staged.ReleaseDevice();
            List<(float[] Video, float[] Audio)> released = Steps(staged, c, null);
            AssertStepBits(Steps(one, c, null), released, c.Name + ", stages after released device copies");
            AssertStepBits(want, released, c.Name + ", stages after released device copies, against the first steps");
            Assert.Equal(2 * one.Compiles, staged.Compiles);
        }

        // Each weight leaf on exactly one stage: the first holds the patch projections and the token refiner, the last the heads.
        using var file = new MiniMaxH3VideoFixture(MiniMaxH3ImageFixture.BF16);
        using var weights = new MiniMaxH3DenoiserWeights(file.DitPath, BackendType.Cpu);
        long[] blockBytes = weights.BlockBytes();
        Assert.Equal(MiniMaxH3VideoFixture.Blocks, blockBytes.Length);
        var first = weights.WeightsOf(new DenoiserSpan(0, 1, IsFirst: true, IsLast: false));
        var last = weights.WeightsOf(new DenoiserSpan(1, 2, IsFirst: false, IsLast: true));
        Assert.Empty(first.Intersect(last));
        Assert.Equal(weights.All.ToHashSet(), first.Concat(last).ToHashSet());
        Assert.Equal(weights.All.Count, first.Count + last.Count);
        Assert.Contains(weights.AudioPatch.Weight, first);
        Assert.Contains(weights.ConditionProj.Weight, first);
        Assert.Contains(weights.FinalAudioOut.Weight, last);
        Assert.Equal(weights.All.Count, weights.WeightsOf(DenoiserSpan.Whole(MiniMaxH3VideoFixture.Blocks)).Count);
    }

    /// <summary>
    /// The same over two CUDA devices on the direct CUDA executors (one a device, the products F32 as on one device), every
    /// case's steps against one GPU's, plain and with the LoRA plug-in where the case has one: within 1e-5 of the largest
    /// value (the stages run one device's kernels; the residual stream they hand on is F32).
    /// </summary>
    [MultiCudaTheory(2)]
    [MemberData(nameof(CaseNames))]
    public void WholeBlockStagesOnTwoCudaDevicesMatchOneDevice(string name)
    {
        MiniMaxH3VideoGoldens.StepCase c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == name);
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        string manifest = c.Lora ? fx.WriteLora() : null;
        using var first = new CudaAllocator(0);
        using var second = new CudaAllocator(1);
        List<(float[] Video, float[] Audio)> want, wantLora = null;
        int compiles;
        using (var one = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, first))
        using (MiniMaxH3LoraSet set = c.Lora ? LoadLora(manifest, one, BackendType.Cuda) : null)
        {
            want = Steps(one, c, null);
            if (set != null) wantLora = Steps(one, c, set);
            compiles = one.Compiles;
        }
        using var staged = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, devices: new IAllocator[] { first, second });
        Assert.Equal(2, staged.Stages.Count);
        AssertStepsClose(want, Steps(staged, c, null), name + " on two CUDA devices");
        if (c.Lora)
        {
            using MiniMaxH3LoraSet stagedSet = LoadLora(manifest, staged, BackendType.Cuda);
            AssertStepsClose(wantLora, Steps(staged, c, stagedSet), name + " on two CUDA devices with the LoRA plug-in");
        }
        Assert.Equal(2 * compiles, staged.Compiles);   // one program a stage wherever one device compiles one
    }

    /// <summary>
    /// The same over two GPUs of one ggml_cuda context (the layer split's: collectives off), each stage's program compiled
    /// and run with its GPU active and its weights preloaded there: every case's steps against one GPU's, plain and with the
    /// LoRA plug-in where the case has one, within 1e-5 of the largest value.
    /// </summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [MemberData(nameof(CaseNames))]
    public void WholeBlockStagesOnTwoGgmlCudaGpusMatchOneGpu(string name)
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        MiniMaxH3VideoGoldens.StepCase c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == name);
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        string manifest = c.Lora ? fx.WriteLora() : null;
        List<(float[] Video, float[] Audio)> want, wantLora = null;
        int compiles;
        using (var one = new MiniMaxH3Denoiser(fx.DitPath, BackendType.GgmlCuda))
        using (MiniMaxH3LoraSet set = c.Lora ? LoadLora(manifest, one, BackendType.GgmlCuda) : null)
        {
            want = Steps(one, c, null);
            if (set != null) wantLora = Steps(one, c, set);
            compiles = one.Compiles;
        }
        var context = new GgmlContext(new[] { 0, 1 }, GgmlBackendType.Cuda, enableCollectives: false);   // as a layer-split model makes it
        var devices = new IAllocator[] { new GgmlAllocator(context, 0), new GgmlAllocator(context, 1) };
        using var staged = new MiniMaxH3Denoiser(fx.DitPath, BackendType.GgmlCuda, devices: devices);
        Assert.Equal(new[] { 0, 1 }, staged.Stages.Devices);
        Assert.True(staged.Preload() > 0);
        AssertStepsClose(want, Steps(staged, c, null), name + " on two ggml_cuda GPUs");
        if (c.Lora)
        {
            using MiniMaxH3LoraSet stagedSet = LoadLora(manifest, staged, BackendType.GgmlCuda);
            AssertStepsClose(wantLora, Steps(staged, c, stagedSet), name + " on two ggml_cuda GPUs with the LoRA plug-in");
        }
        Assert.Equal(2 * compiles, staged.Compiles);   // one program a stage wherever one device compiles one
    }

    // ================================================================ tensor parallelism (--tp N)

    /// <summary>Whether the tests' degrees split a case's fixture: the Q4_K fixture's attention output projection reads 256
    /// columns, one Q4_K block a row, which two ranks cannot share in whole blocks, so the denoiser refuses it
    /// (<see cref="TensorParallelRefusesWhatDoesNotDivide"/>); the release's 7168 columns split over 2, 4 and 7 GPUs.</summary>
    private static bool Splits(MiniMaxH3VideoGoldens.StepCase c) => c.Projections != MiniMaxH3ImageFixture.Q4_K;

    /// <summary>
    /// Each step's video and audio against one device's: the L2 norm of the difference over the reference's within
    /// <paramref name="l2Bound"/>, and the largest difference over the largest value within <paramref name="maxBound"/>, both
    /// logged. The ranks sum each row-parallel projection's partials in another order than one device's product sums its
    /// columns, so the steps agree to rounding, not bit for bit.
    /// </summary>
    private void AssertRanksClose(List<(float[] Video, float[] Audio)> want, List<(float[] Video, float[] Audio)> got, string what,
        double l2Bound, double maxBound)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            var (video, videoL2) = Compare(want[i].Video, got[i].Video);
            var (audio, audioL2) = Compare(want[i].Audio, got[i].Audio);
            output.WriteLine($"[h3-tp] {what}, step {i + 1}: ranks vs one device: video max {video:E3} (L2 {videoL2:E3}), " +
                             $"audio max {audio:E3} (L2 {audioL2:E3})");
            Assert.True(videoL2 <= l2Bound && audioL2 <= l2Bound,
                $"{what}, step {i + 1}: relative L2 video {videoL2:E3}, audio {audioL2:E3} (bound {l2Bound:E1})");
            Assert.True(video <= maxBound && audio <= maxBound,
                $"{what}, step {i + 1}: largest difference video {video:E3}, audio {audio:E3} (bound {maxBound:E1})");
        }
    }

    /// <summary>
    /// The trunk sharded over a simulated group of <paramref name="ranks"/> host devices (SimulatedDeviceGroup: host
    /// allocators, an exact host all-reduce; the managed executor a rank, serving the wide DiT regions as one device's does):
    /// every splitting case's every step, video and audio, equals one device's to rounding, with and without the LoRA plug-in
    /// (each rank binds its share of the trunk's factors, the token refiner's whole on rank 0) and again after it; each rank
    /// compiles where one device compiles (once a geometry and LoRA set), two all-reduces a block a step, the refiner once a
    /// prompt; released device copies give the ranks' first bytes again. A rank holds its shards, not the whole trunk.
    /// <para>
    /// Measured on the x64 dev container (two and four ranks, every step): BF16 within relative L2 2.3e-7 and a largest
    /// difference of 3.9e-7 of the largest value (with the LoRA plug-in; 2.2e-7 without), Q8_0 within 1.5e-7 and 2.3e-7.
    /// Bounds: 1e-5 for both on the float formats; on Q8_0 2e-4 and 2e-3, since the managed block-format products quantize
    /// their activations and another host's kernels may flip one rounding where the reduction order moves a value across a
    /// code boundary (the one-device lane saw a flip move a step by 7.6e-4, CheckCase).
    /// </para>
    /// </summary>
    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public void TensorParallelRanksMatchOneDeviceOnTheManagedExecutor(int ranks)
    {
        foreach (MiniMaxH3VideoGoldens.StepCase c in MiniMaxH3VideoGoldens.Cases.Where(Splits))
        {
            bool roundsActivations = c.Projections == MiniMaxH3ImageFixture.Q8_0;
            double l2Bound = roundsActivations ? 2e-4 : 1e-5, maxBound = roundsActivations ? 2e-3 : 1e-5;
            using var fx = new MiniMaxH3VideoFixture(c.Projections);
            using var one = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu);
            using var group = new SimulatedDeviceGroup(ranks);
            using var tp = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, tensorParallel: group);
            Assert.Equal(ranks, tp.Ranks.Count);
            Assert.Null(tp.Stages);
            Assert.True(tp.OnTensorExecutor);
            Assert.True(tp.WeightBytes < one.WeightBytes, "a rank holds its shards of the trunk, not all of it");

            List<(float[] Video, float[] Audio)> want = Steps(one, c, null);
            int reduces = group.AllReduceCount;
            List<(float[] Video, float[] Audio)> first = Steps(tp, c, null);
            AssertRanksClose(want, first, $"{c.Name} over {ranks} ranks", l2Bound, maxBound);
            Assert.Equal(2 * 2 * MiniMaxH3VideoFixture.Blocks, group.AllReduceCount - reduces);   // two steps, two a block
            // One program a rank wherever one device compiles one (a geometry), replayed by the steps that follow.
            Assert.InRange(one.Compiles, 1, 2);
            Assert.Equal(ranks * one.Compiles, tp.Compiles);
            Assert.Equal(1, one.Refinements);
            Assert.Equal(one.Refinements, tp.Refinements);

            if (c.Lora)
            {
                string manifest = fx.WriteLora();
                using MiniMaxH3LoraSet oneSet = LoadLora(manifest, one, BackendType.Cpu);
                using MiniMaxH3LoraSet tpSet = LoadLora(manifest, tp, BackendType.Cpu);
                AssertRanksClose(Steps(one, c, oneSet), Steps(tp, c, tpSet), $"{c.Name} over {ranks} ranks with the LoRA plug-in",
                    l2Bound, maxBound);
                Assert.Same(tpSet, tp.BoundLora);
                Assert.True(tp.RankLoraCopies > 0, "the ranks' shares of fc1's up and of out's and fc2's downs are copies");
                Assert.Equal(ranks * one.Compiles, tp.Compiles);
                Assert.Equal(one.Refinements, tp.Refinements);
                // Without the adapters again: no rank keeps the set's plan, factors or shares.
                AssertRanksClose(Steps(one, c, null), Steps(tp, c, null), $"{c.Name} over {ranks} ranks after the LoRA plug-in",
                    l2Bound, maxBound);
                Assert.Null(tp.BoundLora);
                Assert.Equal(0, tp.RankLoraCopies);
                Assert.Equal(ranks * one.Compiles, tp.Compiles);
            }

            // Released device copies (every rank's plan with them): the next steps compile each rank again, to the same bytes.
            one.ReleaseDevice();
            tp.ReleaseDevice();
            List<(float[] Video, float[] Audio)> released = Steps(tp, c, null);
            AssertRanksClose(Steps(one, c, null), released, $"{c.Name} over {ranks} ranks after released device copies", l2Bound,
                maxBound);
            AssertStepBits(first, released, $"{c.Name} over {ranks} ranks after released device copies, against the first steps");
            Assert.Equal(ranks * one.Compiles, tp.Compiles);
        }
    }

    /// <summary>
    /// A rank's weights: the trunk's projections are its shares (qkv and fc1 its rows of each stacked part, out and fc2 its
    /// input columns, each rank's shards together the whole projection's bytes), every other leaf is the denoiser's own, the
    /// token refiner's on rank 0 only; a rank's share of a stacked projection is the rank's rows of each part, in part order.
    /// </summary>
    [Fact]
    public void TensorParallelRanksHoldTheirShardsOfEachBlock()
    {
        using var fx = new MiniMaxH3VideoFixture(MiniMaxH3ImageFixture.BF16);
        using var group = new SimulatedDeviceGroup(2);
        using var tp = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, tensorParallel: group);
        using var weights = new MiniMaxH3DenoiserWeights(fx.DitPath, BackendType.Cpu);
        List<MiniMaxH3DitWeight> rank0 = tp.RankWeights(0), rank1 = tp.RankWeights(1);
        Assert.Contains(rank0, w => w.Ne0 == MiniMaxH3VideoFixture.Inner / 2 && w.Ne1 == MiniMaxH3VideoFixture.Hidden);   // out's columns
        Assert.Contains(rank0, w => w.Ne0 == MiniMaxH3VideoFixture.Hidden && w.Ne1 == 3 * MiniMaxH3VideoFixture.Inner / 2);   // qkv's rows
        Assert.Contains(rank0, w => w.Ne0 == MiniMaxH3VideoFixture.Hidden && w.Ne1 == MiniMaxH3VideoFixture.Ffn);   // fc1's rows of each half
        Assert.Contains(rank0, w => w.Ne0 == MiniMaxH3VideoFixture.Ffn / 2 && w.Ne1 == MiniMaxH3VideoFixture.Hidden);   // fc2's columns
        long refiner = weights.Refiner.SelectMany(MiniMaxH3DenoiserWeights.Of).Sum(w => w.Bytes) + weights.ConditionProj.Weight.Bytes +
                       (weights.ConditionProj.Bias?.Bytes ?? 0) + (weights.RefinerFinalNorm?.Bytes ?? 0);
        // Rank 0 holds every weight but the other rank's half of the trunk's projections; rank 1 the same but the refiner.
        Assert.Equal(weights.Bytes - TrunkProjectionBytes(weights) / 2, rank0.Sum(w => w.Bytes));
        Assert.Equal(rank0.Sum(w => w.Bytes) - refiner, rank1.Sum(w => w.Bytes));
        Assert.Equal(rank0.Sum(w => w.Bytes), tp.WeightBytes);

        // Rank 1's qkv share is the rows of its heads of q, then of k, then of v; its fc1 share its rows of gate, then of value.
        MiniMaxH3DitWeight wholeQkv = weights.Blocks[0].Qkv.Weight, wholeFc1 = weights.Blocks[0].Fc1.Weight;
        MiniMaxH3DitWeight qkv = rank1.Single(w => w.Ne1 == 3 * MiniMaxH3VideoFixture.Inner / 2 && w.Bytes == wholeQkv.Bytes / 2 &&
            Rows(w, 0).SequenceEqual(Rows(wholeQkv, MiniMaxH3VideoFixture.Inner / 2)));
        long half = MiniMaxH3VideoFixture.Inner / 2;
        for (int part = 0; part < 3; part++)
            Assert.Equal(Rows(wholeQkv, part * MiniMaxH3VideoFixture.Inner + half, half), Rows(qkv, part * half, half));
        MiniMaxH3DitWeight fc1 = rank1.Single(w => w.Ne1 == MiniMaxH3VideoFixture.Ffn && w.Ne0 == MiniMaxH3VideoFixture.Hidden &&
            Rows(w, 0).SequenceEqual(Rows(wholeFc1, MiniMaxH3VideoFixture.Ffn / 2)));
        for (int part = 0; part < 2; part++)
            Assert.Equal(Rows(wholeFc1, part * MiniMaxH3VideoFixture.Ffn + MiniMaxH3VideoFixture.Ffn / 2, MiniMaxH3VideoFixture.Ffn / 2),
                Rows(fc1, part * MiniMaxH3VideoFixture.Ffn / 2, MiniMaxH3VideoFixture.Ffn / 2));
    }

    /// <summary>The bytes of the trunk's qkv, out, fc1 and fc2 (the projections the ranks shard).</summary>
    private static long TrunkProjectionBytes(MiniMaxH3DenoiserWeights w) =>
        w.Blocks.Sum(b => b.Qkv.Weight.Bytes + b.Out.Weight.Bytes + b.Fc1.Weight.Bytes + b.Fc2.Weight.Bytes);

    /// <summary>Rows [<paramref name="first"/>, + <paramref name="count"/>) of a projection's bytes (one row if count is 1).</summary>
    private static unsafe byte[] Rows(MiniMaxH3DitWeight w, long first, long count = 1)
    {
        long row = w.Bytes / w.Ne1;
        return new ReadOnlySpan<byte>((byte*)w.Data + first * row, checked((int)(count * row))).ToArray();
    }

    /// <summary>
    /// What tensor parallelism refuses, at load, with the reason: a degree that does not divide the heads (every rank holds
    /// whole heads), one whose share of a row-parallel projection's input columns cuts its quant blocks (the Q4_K fixture's
    /// 256 columns over two ranks), the hybrid (tensor-parallel stages), and a GGML backend (its ranked runner takes no DiT
    /// regions). The release's shapes (56 heads of 128, FFN 14336): Q4_K splits over 2, 4 and 7 GPUs, not 8 (896 columns a
    /// GPU), Q8_0 over 8; 3 GPUs divide no head count of 56.
    /// </summary>
    [Fact]
    public void TensorParallelRefusesWhatDoesNotDivide()
    {
        using var fx = new MiniMaxH3VideoFixture(MiniMaxH3ImageFixture.BF16);
        using (var three = new SimulatedDeviceGroup(3))
        {
            var heads = Assert.Throws<NotSupportedException>(() => new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, tensorParallel: three));
            Assert.Contains("--tp 3", heads.Message, StringComparison.Ordinal);
            Assert.Contains($"{MiniMaxH3VideoFixture.Heads} heads", heads.Message, StringComparison.Ordinal);
        }
        using var two = new SimulatedDeviceGroup(2);
        using (var q4 = new MiniMaxH3VideoFixture(MiniMaxH3ImageFixture.Q4_K))
        {
            var blocks = Assert.Throws<NotSupportedException>(() => new MiniMaxH3Denoiser(q4.DitPath, BackendType.Cpu, tensorParallel: two));
            Assert.Contains("out_proj", blocks.Message, StringComparison.Ordinal);
            Assert.Contains("Q4_K", blocks.Message, StringComparison.Ordinal);
            Assert.Contains("--tp 2", blocks.Message, StringComparison.Ordinal);
        }
        var devices = new IAllocator[] { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
        var hybrid = Assert.Throws<NotSupportedException>(() =>
            new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, devices: devices, tensorParallel: two));
        Assert.Contains("--layer-split or --tp", hybrid.Message, StringComparison.Ordinal);
        var ggml = Assert.Throws<NotSupportedException>(() => new MiniMaxH3Denoiser(fx.DitPath, BackendType.GgmlCpu, tensorParallel: two));
        Assert.Contains("ranked graph runner", ggml.Message, StringComparison.Ordinal);
        // A group of one is one device.
        using (var single = new SimulatedDeviceGroup(1))
        using (var dit = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cpu, tensorParallel: single))
            Assert.Null(dit.Ranks);

        var release = new MiniMaxH3Config
        {
            HiddenSize = 5376, NumLayers = 50, NumAttentionHeads = 56, AttentionHeadDim = 128, FfnHiddenSize = 14336,
        };
        IEnumerable<(string, int, long)> RowParallel(int format) => Enumerable.Range(0, release.NumLayers).SelectMany(i => new[]
        {
            ($"blocks.{i}.attn.out_proj.weight", format, (long)release.AttentionInnerDim),
            ($"blocks.{i}.mlp.fc2.weight", format, (long)release.FfnHiddenSize),
        });
        foreach (int degree in new[] { 2, 4, 7 })
            Assert.Null(MiniMaxH3Denoiser.TensorParallelRefusal(release, degree, RowParallel(MiniMaxH3ImageFixture.Q4_K)));
        string eight = MiniMaxH3Denoiser.TensorParallelRefusal(release, 8, RowParallel(MiniMaxH3ImageFixture.Q4_K));
        Assert.Contains("896", eight, StringComparison.Ordinal);
        Assert.Contains("Q4_K", eight, StringComparison.Ordinal);
        Assert.Null(MiniMaxH3Denoiser.TensorParallelRefusal(release, 8, RowParallel(MiniMaxH3ImageFixture.Q8_0)));
        Assert.Contains("56 heads", MiniMaxH3Denoiser.TensorParallelRefusal(release, 3, RowParallel(MiniMaxH3ImageFixture.BF16)),
            StringComparison.Ordinal);
        Assert.Null(MiniMaxH3Denoiser.TensorParallelRefusal(release, 1, RowParallel(MiniMaxH3ImageFixture.Q4_K)));
    }

    /// <summary>The model refuses --tp off the direct CUDA backend at load, before its companions are resolved, naming the
    /// backend: a supplied group on cpu does not reach a denoiser.</summary>
    [Fact]
    public void TheModelRefusesTensorParallelismOffCuda()
    {
        using var fx = new MiniMaxH3VideoFixture(MiniMaxH3ImageFixture.BF16);
        var group = new SimulatedDeviceGroup(2);
        var e = Assert.Throws<ModelLoadRefusedException>(() => new MiniMaxH3Model(fx.DitPath, BackendType.Cpu, tpDegree: 2, tpGroup: group));
        Assert.Contains("--backend cuda", e.Message, StringComparison.Ordinal);
        Assert.Contains("--tp 2", e.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The same over two CUDA devices (the direct CUDA backend's group: one tensor executor a device, the partials summed by
    /// the device-ordered peer reduction), every case's steps against one GPU's, plain and with the LoRA plug-in where the
    /// case has one: within relative L2 1e-5 and 1e-4 of the largest value (the products keep their activations in F32 on
    /// both; the ranks sum the row-parallel partials in another order). The Q4_K fixture is refused (it does not split over
    /// two in whole blocks). Every rank compiles once a geometry; its weights' copies stay on its device after a step and go
    /// with the denoiser.
    /// </summary>
    [MultiCudaTheory(2)]
    [MemberData(nameof(CaseNames))]
    public void TensorParallelOnTwoCudaDevicesMatchesOneDevice(string name)
    {
        MiniMaxH3VideoGoldens.StepCase c = MiniMaxH3VideoGoldens.Cases.Single(x => x.Name == name);
        using var fx = new MiniMaxH3VideoFixture(c.Projections);
        using var group = new TensorSharp.Cuda.TensorParallelGroup(2);
        if (!Splits(c))
        {
            var refused = Assert.Throws<NotSupportedException>(() =>
                new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, group.GetAllocator(0), tensorParallel: group));
            Assert.Contains("Q4_K", refused.Message, StringComparison.Ordinal);
            return;
        }
        string manifest = c.Lora ? fx.WriteLora() : null;
        List<(float[] Video, float[] Audio)> want, wantLora = null;
        int compiles;
        using (var one = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, group.GetAllocator(0)))
        using (MiniMaxH3LoraSet set = c.Lora ? LoadLora(manifest, one, BackendType.Cuda) : null)
        {
            want = Steps(one, c, null);
            if (set != null) wantLora = Steps(one, c, set);
            compiles = one.Compiles;
        }
        var tp = new MiniMaxH3Denoiser(fx.DitPath, BackendType.Cuda, group.GetAllocator(0), tensorParallel: group);
        try
        {
            Assert.Equal(2, tp.Ranks.Count);
            AssertRanksClose(want, Steps(tp, c, null), name + " over two CUDA devices", 1e-5, 1e-4);
            if (c.Lora)
            {
                using MiniMaxH3LoraSet tpSet = LoadLora(manifest, tp, BackendType.Cuda);
                AssertRanksClose(wantLora, Steps(tp, c, tpSet), name + " over two CUDA devices with the LoRA plug-in", 1e-5, 1e-4);
            }
            Assert.Equal(2 * compiles, tp.Compiles);   // one program a rank wherever one device compiles one
            IReadOnlyList<IntPtr> keys = tp.DeviceCopyKeysForTest();
            int Resident(int rank) => keys.Count(k => CudaQuantizedOps.TryGetResidentDevicePtr(group.GetCudaAllocator(rank), k, out _, out _));
            Assert.True(Resident(0) > 0 && Resident(1) > 0, "each rank's weights stay resident on its device");
            tp.Dispose();
            Assert.Equal(0, Resident(0));
            Assert.Equal(0, Resident(1));
        }
        finally
        {
            tp.Dispose();
        }
    }

    /// <summary>The family's contract: the shared diffusion runtime's, with layer split on cuda and ggml_cuda (whole-block
    /// stages) and tensor parallelism on cuda (the ranks), both proven by the tests above, which the gate lets through there; tensor
    /// parallelism on ggml_cuda and the hybrid are gaps the gate refuses, naming the family; a backend of one device refuses
    /// both, and distributed groups are refused.</summary>
    [Fact]
    public void TensorParallelOnCudaAndLayerSplitPassTheirGates()
    {
        Assert.True(ModelArchitectureRegistry.TryGet("minimax-h3", out var arch));
        Assert.Equal(ServingImplementation.SharedDiffusionRuntime, arch.Serving.Implementation);
        Assert.Equal(new[] { BackendType.Cuda, BackendType.GgmlCuda }, arch.Serving.LayerSplit.Backends);
        Assert.Equal(new[] { BackendType.Cuda }, arch.Serving.TensorParallel.Backends);
        Assert.Empty(arch.Serving.HybridParallel.Backends);
        Assert.False(arch.SupportsDistributedTensorParallel);
        Assert.Contains("minimax-h3", arch.Serving.TensorParallel.GapReason, StringComparison.Ordinal);
        foreach (BackendType backend in new[] { BackendType.Cuda, BackendType.GgmlCuda })
        {
            ITensorParallelGroup group = null;
            Assert.Equal(1, ModelBase.ResolveTensorParallelSupport(arch, backend, 1, ref group, out int split, 2));
            Assert.Equal(2, split);
            if (backend == BackendType.Cuda)
            {
                Assert.Equal(2, ModelBase.ResolveTensorParallelSupport(arch, backend, 2, ref group, out int noSplit));
                Assert.Equal(1, noSplit);
            }
            else
            {
                var tp = Assert.Throws<NotSupportedException>(() => ModelBase.ResolveTensorParallelSupport(arch, backend, 2, ref group, out _));
                Assert.Contains("--tp 2", tp.Message, StringComparison.Ordinal);
                Assert.Contains("minimax-h3", tp.Message, StringComparison.Ordinal);
            }
            var hybrid = Assert.Throws<NotSupportedException>(() =>
                ModelBase.ResolveTensorParallelSupport(arch, backend, 2, ref group, out _, 2));
            Assert.Contains("--layer-split 2", hybrid.Message, StringComparison.Ordinal);
            Assert.Null(group);
        }
        ITensorParallelGroup none = null;
        var oneDevice = Assert.Throws<NotSupportedException>(() =>
            ModelBase.ResolveTensorParallelSupport(arch, BackendType.GgmlCpu, 1, ref none, out _, 2));
        Assert.Contains("layer-split", oneDevice.Message, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedLayerSplit(arch, BackendType.GgmlCuda));
        Assert.Throws<NotSupportedException>(() => ModelBase.ValidateDistributedTensorParallelism(arch, BackendType.GgmlCuda));
    }

    /// <summary>The retired pipeline, DiTs and native forward are gone: the runtime's denoiser is the only one.</summary>
    [Fact]
    public void TheRetiredDenoisersAreGone()
    {
        System.Reflection.Assembly models = typeof(ModelBase).Assembly;
        foreach (string gone in new[]
        {
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3DiT",
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3DirectDiT",
            "TensorSharp.Models.MiniMaxH3.MiniMaxH3Pipeline",
        })
            Assert.Null(models.GetType(gone));
        Assert.Null(typeof(TensorSharp.GGML.GgmlBasicOps).GetMethod("TryMiniMaxH3DitForward"));
        Assert.True(typeof(IDenoiser).IsAssignableFrom(typeof(MiniMaxH3Denoiser)));
    }
}
