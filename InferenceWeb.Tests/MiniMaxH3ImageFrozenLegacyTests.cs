// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image on the diffusion runtime against the legacy oracle's numbers, frozen per lane before stage S10 deleted the
/// legacy denoiser and decoder (<see cref="MiniMaxH3ImageGoldens"/>): the runtime denoiser's six steps and the runtime
/// decoder's pixels on the synthetic fixtures, byte for byte on a lane where the runtime gave the legacy bytes when they were
/// frozen (ggml_metal on the M5 Pro, ggml_cuda on the RTX PRO 4500), within the bound elsewhere; the managed executor within
/// its bound; the step's host tables (C#).
/// </summary>
public sealed class MiniMaxH3ImageFrozenLegacyTests(ITestOutputHelper output)
{
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    /// <summary>Off the lane a case was frozen on, the bound on its largest sampled difference over the largest legacy value,
    /// by the projections' format. Measured at the freeze (ggml d9fe7bff): between the four lanes' legacy outputs (ggml_cpu
    /// on the M5 Pro and on an x64 EPYC, ggml_metal on the M5 Pro, ggml_cuda on the RTX PRO 4500) at most 5.8e-4 (F32),
    /// 6.1e-4 (BF16), 1.5e-3 (Q8_0), 2.6e-3 (Q4_K); the runtime against legacy on ARM64 ggml_cpu (main's products) at most
    /// 5.2e-5, 5.5e-4, 1.6e-3, 2.5e-3. The bound is about twice their sum or more.</summary>
    private static double Bound(int format) => format is Q8_0 or Q4_K ? 1e-2 : 3e-3;

    private static void Pin(BackendType backend) => GgmlBasicOps.EnsureBackendAvailable(TestGates.GgmlBackendTypeOf(backend));

    public static IEnumerable<object[]> DenoiserCases() => new (string, int)[]
    {
        ("TextToImage", F32), ("TextToImage", BF16), ("TextToImage", Q8_0), ("TextToImage", Q4_K),
        ("WithReferences", F32), ("WithReferences", BF16), ("WithReferences", Q8_0), ("WithReferences", Q4_K),
        ("Wide", BF16), ("Wide", Q4_K), ("ManySegments", BF16), ("ManySegments", Q4_K),
    }.Select(c => new object[] { c.Item1, c.Item2 });

    private void VerifyDenoiser(BackendType backend, string request, int format)
    {
        Pin(backend);
        string caseName = MiniMaxH3ImageGoldens.DenoiserCase(request, format);
        using var live = new MiniMaxH3ImageFixture(format);
        string sha = MiniMaxH3ImageGoldens.FileSha(live.DitPath);
        // ggml's Q4_K quantizer rounds otherwise on a CPU or build no lane was frozen on: rebuild the fixture from the
        // reference lane's machine's bytes, and hold the runtime to that lane's outputs within the bound.
        using var pinned = format == Q4_K && !MiniMaxH3ImageGoldens.Froze(caseName, sha)
            ? new MiniMaxH3ImageFixture(format, pinnedQ4K: true) : null;
        if (pinned != null)
        {
            output.WriteLine($"{caseName}: this machine quantizes the fixture to {sha}, which no lane froze; using the pinned Q4_K bytes");
            sha = MiniMaxH3ImageGoldens.FileSha(pinned.DitPath);
        }
        MiniMaxH3ImageFixture fx = pinned ?? live;
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
        var (steps, _) = MiniMaxH3ImageGoldens.Run(family, runtime, MiniMaxH3ImageGoldens.Named(request));
        MiniMaxH3ImageGoldens.Check(caseName, MiniMaxH3ImageGoldens.LaneKey(backend), sha, steps, Bound(format), output.WriteLine);
        // One program for the request's six steps (the head, latent and time embedding are its inputs).
        Assert.Equal(1, runtime.Compiles);
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(DenoiserCases))]
    public void DenoiserStepsAreTheFrozenLegacyStepsOnGgmlCpu(string request, int format) => VerifyDenoiser(BackendType.GgmlCpu, request, format);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(DenoiserCases))]
    public void DenoiserStepsAreTheFrozenLegacyStepsOnGgmlMetal(string request, int format) => VerifyDenoiser(BackendType.GgmlMetal, request, format);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(DenoiserCases))]
    public void DenoiserStepsAreTheFrozenLegacyStepsOnGgmlCuda(string request, int format) => VerifyDenoiser(BackendType.GgmlCuda, request, format);

    public static IEnumerable<object[]> VaeCases() => MiniMaxH3ImageGoldens.VaeCases.Select(c => new object[] { c.Name });

    private void VerifyDecoder(BackendType backend, string name)
    {
        Pin(backend);
        MiniMaxH3ImageGoldens.VaeCase c = MiniMaxH3ImageGoldens.VaeCases.Single(v => v.Name == name);
        using var fx = new MiniMaxH3ImageVaeFixture(c.Blocks);
        float[] latent = MiniMaxH3ImageGoldens.NormalizedLatent(c.LatentH, c.LatentW, c.Seed);
        using var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend) { BatchLimit = name == "vae/split-4" ? 4 : MiniMaxH3ImageVaeCodec.MaxBatch };
        RgbImage got = codec.Decode(latent, c.LatentH, c.LatentW);
        Assert.Equal(c.LatentW * MiniMaxH3ImageVaeCodec.Downsample, got.Width);
        Assert.Equal(c.LatentH * MiniMaxH3ImageVaeCodec.Downsample, got.Height);
        MiniMaxH3ImageGoldens.Check(name, MiniMaxH3ImageGoldens.LaneKey(backend), MiniMaxH3ImageGoldens.FileSha(fx.VaePath),
            new[] { got.Pixels }, 3e-3, output.WriteLine);
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(VaeCases))]
    public void DecodedPixelsAreTheFrozenLegacyPixelsOnGgmlCpu(string name) => VerifyDecoder(BackendType.GgmlCpu, name);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(VaeCases))]
    public void DecodedPixelsAreTheFrozenLegacyPixelsOnGgmlMetal(string name) => VerifyDecoder(BackendType.GgmlMetal, name);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(VaeCases))]
    public void DecodedPixelsAreTheFrozenLegacyPixelsOnGgmlCuda(string name) => VerifyDecoder(BackendType.GgmlCuda, name);

    /// <summary>The managed executor runs the same program through the regions' reference ops (its own products and
    /// attention): against the legacy steps frozen on the process's GGML lane, every step within a bound about five times the
    /// largest difference measured when legacy was live (F32 9e-5 on ggml_cpu and 6.2e-4 on ggml_metal, BF16 4.3e-4 and
    /// 5.8e-4, Q8_0 1.4e-3 and 1.8e-3).</summary>
    [Theory]
    [InlineData(F32, 3e-3)]
    [InlineData(BF16, 3e-3)]
    [InlineData(Q8_0, 1e-2)]
    public void ManagedExecutorStepsAreWithinTheBoundOfTheFrozenLegacySteps(int format, double bound)
    {
        BackendType ggml = TestGates.PinnedGgmlBackend;
        using var fx = new MiniMaxH3ImageFixture(format);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var managed = new MiniMaxH3ImageDenoiser(fx.DitPath, BackendType.Cpu);
        Assert.IsType<TensorSharp.Models.Graph.ManagedGraphExecutor>(managed.Executor);
        var (steps, _) = MiniMaxH3ImageGoldens.Run(family, managed, MiniMaxH3ImageGoldens.Request.WithReferences);
        // Against the process's GGML lane's frozen steps (or the nearest lane's), never byte for byte: the managed products
        // are not ggml's.
        string caseName = MiniMaxH3ImageGoldens.DenoiserCase("WithReferences", format);
        MiniMaxH3ImageGoldens.Case frozen = MiniMaxH3ImageGoldens.Find(caseName, MiniMaxH3ImageGoldens.LaneKey(ggml), MiniMaxH3ImageGoldens.FileSha(fx.DitPath));
        MiniMaxH3ImageGoldens.CheckAgainst(frozen, caseName, "managed", MiniMaxH3ImageGoldens.FileSha(fx.DitPath), steps, bound, output.WriteLine);
        Assert.Equal(1, managed.Compiles);
    }

    /// <summary>The step's host tables (the AdaLN curve table at the layout's timesteps and the 3-axis partial-RoPE tables of
    /// its float positions) are the legacy DiT's: byte for byte on the runtime identifier they were frozen on, within float
    /// rounding (1e-6 of the largest value) elsewhere.</summary>
    [PinnedGgmlServesFact(MiniMaxH3ImageFamily.ArchitectureId)]
    public void TheStepsHostTablesAreTheFrozenLegacyTables()
    {
        using var fx = new MiniMaxH3ImageFixture(BF16);
        var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
        using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        MiniMaxH3PackedLayout layout = MiniMaxH3ImageGoldens.HostTablesLayout(family);
        float[] te = runtime.TimeEmbedding(layout.Timesteps);
        var (cos, sin) = runtime.Rope((float[])layout.Positions.Clone());
        MiniMaxH3ImageGoldens.Check("host/tables", MiniMaxH3ImageGoldens.HostLaneKey(), MiniMaxH3ImageGoldens.FileSha(fx.DitPath),
            new[] { te, cos, sin }, 1e-6, output.WriteLine);
        Assert.Same(cos, runtime.Rope((float[])layout.Positions.Clone()).Cos);   // kept for the request's steps
    }

    /// <summary>Every lane's file holds every case, from the same fixtures (one SHA-256 a case across the lanes).</summary>
    [Fact]
    public void EveryFrozenLaneHoldsEveryCaseFromTheSameFixtures()
    {
        string[] denoiser = DenoiserCases().Select(o => MiniMaxH3ImageGoldens.DenoiserCase((string)o[0], (int)o[1])).ToArray();
        string[] decoder = MiniMaxH3ImageGoldens.VaeCases.Select(c => c.Name).ToArray();
        Assert.Contains(MiniMaxH3ImageGoldens.Lanes, l => l.StartsWith("ggml_cpu|osx-arm64|", StringComparison.Ordinal));
        Assert.Contains(MiniMaxH3ImageGoldens.Lanes, l => l.StartsWith("ggml_metal|", StringComparison.Ordinal));
        Assert.Contains(MiniMaxH3ImageGoldens.Lanes, l => l.StartsWith("ggml_cuda|", StringComparison.Ordinal));
        // A case's fixture bytes are one per runtime identifier's architecture (the Q4_K quantizer's floats round otherwise
        // on x64 than on ARM64).
        var fixtureOf = new Dictionary<(string, string), string>();
        foreach (string lane in MiniMaxH3ImageGoldens.Lanes.Where(l => l.StartsWith("ggml_", StringComparison.Ordinal)))
            foreach (string name in denoiser.Concat(decoder))
            {
                MiniMaxH3ImageGoldens.Case c = MiniMaxH3ImageGoldens.Find(name, lane);
                Assert.True(c != null && c.Lane == lane, $"{lane} has no {name}");
                string arch = lane.Split('|')[1].Split('-').Last();
                if (fixtureOf.TryGetValue((name, arch), out string sha)) Assert.Equal(sha, c.FixtureSha256);
                else fixtureOf[(name, arch)] = c.FixtureSha256;
            }
    }
}
