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
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The Muse-Glimmer / Muse Spark vision tower as a graph program (<see cref="MuseGlimmerVisionTower"/>, stage S4 of
/// docs/design/muse-glimmer-unified.md) against the legacy per-op tower (<see cref="MuseGlimmerVisionEncoder"/>, its oracle)
/// on the same executor, over a synthetic mmproj laid out as the real one (<see cref="MuseGlimmerVisionSynthetic"/>: Q8_0
/// products, windows of 4 x 4 patches, layers 0-2 windowed and 3-4 global): square and non-square grids from one window to
/// six, F16 and F32 keys and values on GGML, several images on one tower, a large image split into programs, attention in
/// pieces.
/// </summary>
public sealed class MuseGlimmerVisionTowerTests(ITestOutputHelper output) : IDisposable
{
    /// <summary>
    /// The bound on each image's relative L2 distance and largest difference over the largest legacy value, against legacy on
    /// the same executor, over the F32-product fixture. The two towers run different kernels for the same arithmetic: GGML's
    /// flash attention reads F16 keys and values (llama.cpp's mtmd and legacy's fused CUDA block do too; legacy's per-op GGML
    /// attention is F32 products), the executors' rotary and exact GELU stand in for legacy's host loops (its GELU is a 1.5e-7
    /// erf approximation), and reduction orders differ. Measured on the M5 Pro (worst over the grids): managed 6.1e-7, ggml_cpu
    /// 1.3e-4 (3.9e-7 with F32 keys and values), ggml_metal 5.7e-4 (4.5e-4), MLX 6.1e-4. Legacy's own spread between those
    /// executors is 4.1e-3 (MLX against managed; Metal against managed 1.8e-3): the bound, 2e-3, is under it and three times
    /// the worst tower distance. <see cref="TheBoundIsWithinLegacysOwnSpreadBetweenExecutors"/> holds it there.
    /// </summary>
    internal const double ParityBound = 2e-3;

    /// <summary>
    /// The bound over the Q8_0-product fixture (the real mmproj's format), which checks that the tower reads block weights on
    /// every executor, against legacy on the managed backend (which multiplies F32 copies of the weights): the executors whose
    /// Q8_0 products quantize their activations (ggml-cpu, ggml-cuda, the managed executor's integer dot) leave 2.2e-2 between
    /// legacy's own runs on this fixture's random weights (legacy managed against legacy ggml_cpu), and the tower is within
    /// 2.1e-2 of managed legacy on ggml_cuda and ggml_vulkan, 3.0e-2 of legacy on ggml_cpu; a wrong block layout or
    /// orientation is O(1).
    /// </summary>
    internal const double Q8ProductBound = 5e-2;

    /// <summary>Patch grids (across, down): one window (4 x 4), four (8 x 8), six with clipped edges (6 x 10, 12 x 6), a strip
    /// (10 x 2: three windows) and a column (2 x 6: two).</summary>
    internal static readonly (int W, int H)[] Grids = { (4, 4), (8, 8), (6, 10), (10, 2), (2, 6), (12, 6) };

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-vtower-" + Guid.NewGuid().ToString("N"));
    private string _mmproj;

    private string Mmproj
    {
        get
        {
            if (_mmproj != null) return _mmproj;
            Directory.CreateDirectory(_dir);
            return _mmproj = MuseGlimmerVisionSynthetic.Write(Path.Combine(_dir, "mmproj-synthetic.gguf"));
        }
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>One executor's allocator (and the cleanup GGML's shared caches need between lanes).</summary>
    internal sealed class Lane : IDisposable
    {
        private readonly IDisposable _owner;

        public Lane(BackendType backend)
        {
            Backend = backend;
            switch (backend)
            {
                case BackendType.Cpu:
                    Allocator = new CpuAllocator(BlasEnum.DotNet);
                    break;
                case BackendType.Mlx:
                    var mlx = new MlxAllocator();
                    _owner = mlx;
                    Allocator = mlx;
                    break;
                case BackendType.Cuda:
                    var cuda = new CudaAllocator();
                    _owner = cuda;
                    Allocator = cuda;
                    break;
                default:
                    GgmlBackendType g = backend switch
                    {
                        BackendType.GgmlMetal => GgmlBackendType.Metal,
                        BackendType.GgmlCuda => GgmlBackendType.Cuda,
                        BackendType.GgmlVulkan => GgmlBackendType.Vulkan,
                        _ => GgmlBackendType.Cpu,
                    };
                    Allocator = new GgmlAllocator(new GgmlContext([0], g), 0);
                    Ggml = true;
                    break;
            }
        }

        public BackendType Backend { get; }
        public IAllocator Allocator { get; }
        public bool Ggml { get; }

        public void Dispose()
        {
            _owner?.Dispose();
            if (Ggml)
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
        }
    }

    internal static float[] Legacy(Lane lane, string mmproj, float[] pixels, int width, int height)
    {
        using var encoder = new MuseGlimmerVisionEncoder(mmproj, lane.Allocator);
        return encoder.EncodeToHostArray(pixels, width, height, out _);
    }

    internal static MuseGlimmerVisionTower Tower(Lane lane, string mmproj)
    {
        using var gguf = new GgufFile(mmproj);
        return new MuseGlimmerVisionTower(gguf, lane.Allocator, lane.Backend);
    }

    internal static (double RelL2, double RelMax) Distance(float[] reference, float[] actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        double d2 = 0, r2 = 0, dmax = 0, rmax = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = (double)actual[i] - reference[i];
            Assert.False(double.IsNaN(d) || double.IsInfinity(d), $"value {i} is not finite ({actual[i]})");
            d2 += d * d;
            r2 += (double)reference[i] * reference[i];
            dmax = Math.Max(dmax, Math.Abs(d));
            rmax = Math.Max(rmax, Math.Abs(reference[i]));
        }
        return (Math.Sqrt(d2 / Math.Max(r2, 1e-30)), dmax / Math.Max(rmax, 1e-30));
    }

    private static (int Width, int Height, float[] Pixels) Image((int W, int H) grid, uint seed)
    {
        int width = grid.W * MuseGlimmerVisionSynthetic.Patch, height = grid.H * MuseGlimmerVisionSynthetic.Patch;
        return (width, height, MuseGlimmerVisionSynthetic.Pixels(width, height, seed));
    }

    private string _mmprojF32;

    /// <summary>The fixture with F32 products (the parity gates).</summary>
    private string MmprojF32
    {
        get
        {
            if (_mmprojF32 != null) return _mmprojF32;
            Directory.CreateDirectory(_dir);
            return _mmprojF32 = MuseGlimmerVisionSynthetic.Write(Path.Combine(_dir, "mmproj-synthetic-f32.gguf"),
                products: SyntheticGguf.GgmlType.F32);
        }
    }

    /// <summary>
    /// Every grid through legacy and the tower on <paramref name="backend"/> over the F32-product fixture: within
    /// <see cref="ParityBound"/> of legacy. On GGML the tower also runs with F32 keys and values, which takes it to legacy's
    /// per-op attention (the F16 keys and values are the only difference there).
    /// </summary>
    private void Verify(BackendType backend)
    {
        using var lane = new Lane(backend);
        using MuseGlimmerVisionTower tower = Tower(lane, MmprojF32);
        using MuseGlimmerVisionTower f32Kv = Tower(lane, MmprojF32);
        f32Kv.Graph.AttentionHalfKv = false;
        output.WriteLine($"executor {tower.Graph.Executor.GetType().Name} ({backend}); F16 keys and values: {tower.Graph.AttentionHalfKv}");
        double worst = 0, worstF32Kv = 0;
        uint seed = 1;
        foreach (var grid in Grids)
        {
            var (w, h, px) = Image(grid, seed++);
            float[] reference = Legacy(lane, MmprojF32, px, w, h);
            float[] a = tower.EncodeToHost(px, w, h, out int tokens);
            float[] c = f32Kv.EncodeToHost(px, w, h, out _);
            Assert.Equal(grid.W * grid.H / 4, tokens);
            Assert.Equal(tokens * 48, a.Length);
            var da = Distance(reference, a);
            var dc = Distance(reference, c);
            int windows = tower.GeometryOf(grid.W, grid.H).WindowCount;
            output.WriteLine($"  grid {grid.W}x{grid.H} ({windows} window{(windows == 1 ? "" : "s")}): relL2 {da.RelL2:E2} max {da.RelMax:E2}; " +
                $"F32 keys and values {Math.Max(dc.RelL2, dc.RelMax):E2}");
            worst = Math.Max(worst, Math.Max(da.RelL2, da.RelMax));
            worstF32Kv = Math.Max(worstF32Kv, Math.Max(dc.RelL2, dc.RelMax));
        }
        output.WriteLine($"  worst vs legacy {worst:E3} (bound {ParityBound:E1}); with F32 keys and values {worstF32Kv:E3}");
        Assert.True(worst <= ParityBound, $"{backend}: {worst:E3} from the legacy tower (bound {ParityBound:E1})");
        Assert.True(worstF32Kv <= ParityBound, $"{backend}: with F32 keys and values {worstF32Kv:E3} from the legacy tower");
    }

    /// <summary>
    /// The Q8_0-product fixture (the real mmproj's format) on <paramref name="backend"/>: within <see cref="Q8ProductBound"/> of
    /// legacy on the managed backend. Legacy's own tower on the backend is measured against the same reference and logged, not
    /// held: on ggml_cuda and ggml_vulkan its ggml tower is O(1) off on this fixture's non-square grids (0.44-1.11 on an RTX PRO
    /// 4500 Blackwell, where the graph tower is within 2.1e-2), a defect of the legacy tower, which stage S9 deletes. On direct
    /// CUDA the tower multiplies F32 copies of the weights' values (<see cref="TensorSharp.Graph.GraphProgram.FloatProducts"/>),
    /// managed legacy's arithmetic, so it is held to <see cref="ParityBound"/> there (6.4e-7 measured).
    /// </summary>
    private void VerifyQ8(BackendType backend)
    {
        using var managed = new Lane(BackendType.Cpu);
        using var lane = new Lane(backend);
        using MuseGlimmerVisionTower tower = Tower(lane, Mmproj);
        double worst = 0, worstLegacy = 0;
        uint seed = 1;
        foreach (var grid in Grids)
        {
            var (w, h, px) = Image(grid, seed++);
            float[] reference = Legacy(managed, Mmproj, px, w, h);
            var d = Distance(reference, tower.EncodeToHost(px, w, h, out _));
            var l = Distance(reference, Legacy(lane, Mmproj, px, w, h));
            output.WriteLine($"  grid {grid.W}x{grid.H}: tower {Math.Max(d.RelL2, d.RelMax):E2}, legacy on {backend} {Math.Max(l.RelL2, l.RelMax):E2} " +
                "from legacy on the managed backend");
            worst = Math.Max(worst, Math.Max(d.RelL2, d.RelMax));
            worstLegacy = Math.Max(worstLegacy, Math.Max(l.RelL2, l.RelMax));
        }
        output.WriteLine($"Q8_0 products on {backend}: worst {worst:E3} from the managed legacy tower (bound " +
            $"{(backend == BackendType.Cuda ? ParityBound : Q8ProductBound):E1}); " +
            $"legacy's own tower there {worstLegacy:E3} (logged)");
        double bound = backend == BackendType.Cuda ? ParityBound : Q8ProductBound;
        Assert.True(worst <= bound, $"{backend}: with Q8_0 products {worst:E3} from the managed legacy tower (bound {bound:E1})");
    }

    [Fact]
    public void TowerOnTheManagedBackendGivesTheLegacyTowersRows() => Verify(BackendType.Cpu);

    [GgmlFact(BackendType.GgmlCpu)]
    public void TowerOnGgmlCpuGivesTheLegacyTowersRows() => Verify(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TowerOnGgmlMetalGivesTheLegacyTowersRows() => Verify(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void TowerOnGgmlCudaGivesTheLegacyTowersRows() => Verify(BackendType.GgmlCuda);

    [GgmlFact(BackendType.GgmlVulkan)]
    public void TowerOnGgmlVulkanGivesTheLegacyTowersRows() => Verify(BackendType.GgmlVulkan);

    [MlxFact]
    public void TowerOnMlxGivesTheLegacyTowersRows() => Verify(BackendType.Mlx);

    [CudaFact]
    public void TowerOnDirectCudaGivesTheLegacyTowersRows() => Verify(BackendType.Cuda);

    [Fact]
    public void TowerReadsQ8ProductsOnTheManagedBackend() => VerifyQ8(BackendType.Cpu);

    [GgmlFact(BackendType.GgmlCpu)]
    public void TowerReadsQ8ProductsOnGgmlCpu() => VerifyQ8(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TowerReadsQ8ProductsOnGgmlMetal() => VerifyQ8(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void TowerReadsQ8ProductsOnGgmlCuda() => VerifyQ8(BackendType.GgmlCuda);

    [GgmlFact(BackendType.GgmlVulkan)]
    public void TowerReadsQ8ProductsOnGgmlVulkan() => VerifyQ8(BackendType.GgmlVulkan);

    [MlxFact]
    public void TowerReadsQ8ProductsOnMlx() => VerifyQ8(BackendType.Mlx);

    [CudaFact]
    public void TowerReadsQ8ProductsOnDirectCuda() => VerifyQ8(BackendType.Cuda);

    /// <summary>
    /// <see cref="ParityBound"/> against the legacy tower's own spread on the F32-product fixture: legacy on each executor this
    /// process builds against legacy on the managed backend. With MLX in the process (the Mac, where the bound was set), the
    /// bound is no looser than that spread; elsewhere legacy's executors agree within three times the bound (a CUDA box's
    /// spread is recorded, not assumed).
    /// </summary>
    [Fact]
    public void TheBoundIsWithinLegacysOwnSpreadBetweenExecutors()
    {
        var backends = new List<BackendType> { TestGates.PinnedGgmlBackend };
        if (TestGates.MlxSkip == null) backends.Add(BackendType.Mlx);
        if (TestGates.CudaSkip == null) backends.Add(BackendType.Cuda);
        using var managed = new Lane(BackendType.Cpu);
        double spread = 0;
        foreach (BackendType backend in backends)
        {
            using var lane = new Lane(backend);
            uint seed = 1;
            double worst = 0;
            foreach (var grid in Grids)
            {
                var (w, h, px) = Image(grid, seed++);
                var d = Distance(Legacy(managed, MmprojF32, px, w, h), Legacy(lane, MmprojF32, px, w, h));
                worst = Math.Max(worst, Math.Max(d.RelL2, d.RelMax));
            }
            output.WriteLine($"legacy on {backend} vs legacy on the managed backend: worst {worst:E3}");
            spread = Math.Max(spread, worst);
        }
        output.WriteLine($"legacy's spread {spread:E3}; the bound {ParityBound:E1}");
        if (backends.Contains(BackendType.Mlx))
            Assert.True(ParityBound <= spread, $"the bound {ParityBound:E1} is looser than legacy's own spread {spread:E3}");
        else
            Assert.True(spread <= 3 * ParityBound, $"legacy's executors are {spread:E3} apart, past three times the bound {ParityBound:E1}");
    }

    /// <summary>Several images on one tower: each geometry's program is compiled once and replayed, a replay gives the rows a
    /// fresh tower gives, and another geometry in between changes nothing.</summary>
    [Fact]
    public void SeveralImagesOnOneTowerReplayTheirPrograms()
    {
        foreach (BackendType backend in new[] { BackendType.Cpu, TestGates.PinnedGgmlBackend })
        {
            using var lane = new Lane(backend);
            using MuseGlimmerVisionTower tower = Tower(lane, MmprojF32);
            var (wa, ha, pa) = Image((6, 10), 11);
            var (wb, hb, pb) = Image((8, 8), 12);
            var (wc, hc, pc) = Image((6, 10), 13);
            float[] a1 = tower.EncodeToHost(pa, wa, ha, out _);
            float[] b = tower.EncodeToHost(pb, wb, hb, out _);
            float[] a2 = tower.EncodeToHost(pa, wa, ha, out _);
            float[] c = tower.EncodeToHost(pc, wc, hc, out _);
            using MuseGlimmerVisionTower fresh = Tower(lane, MmprojF32);
            float[] cFresh = fresh.EncodeToHost(pc, wc, hc, out _);
            float[] bFresh = fresh.EncodeToHost(pb, wb, hb, out _);
            output.WriteLine($"{backend}: {tower.Graph.Builds} builds, {tower.Graph.Replays} replays over 4 images");
            Assert.Equal(a1, a2);
            Assert.Equal(cFresh, c);
            Assert.Equal(bFresh, b);
            Assert.NotEqual(a1, c);
            if (tower.Graph.Executor.Capabilities.Replays) Assert.Equal(2, tower.Graph.Builds);
        }
    }

    /// <summary>A large image with a yield callback runs as programs over consecutive layer ranges, the callback between
    /// them, and gives the rows of one program.</summary>
    [Fact]
    public void ALargeImageSplitIntoProgramsGivesTheRowsOfOne()
    {
        foreach (BackendType backend in new[] { BackendType.Cpu, TestGates.PinnedGgmlBackend })
        {
            using var lane = new Lane(backend);
            using MuseGlimmerVisionTower tower = Tower(lane, MmprojF32);
            var (w, h, px) = Image((12, 6), 21);
            float[] one = tower.EncodeToHost(px, w, h, out _, betweenParts: null);
            Assert.Equal(1, tower.LastParts);
            tower.YieldPatches = 16;
            int yields = 0;
            float[] split = tower.EncodeToHost(px, w, h, out _, betweenParts: () => yields++);
            Assert.Equal(5, tower.LastParts);
            Assert.Equal(4, yields);
            var d = Distance(one, split);
            output.WriteLine($"{backend}: 5 programs vs one: relL2 {d.RelL2:E2}, max {d.RelMax:E2}");
            Assert.True(Math.Max(d.RelL2, d.RelMax) <= 1e-6, $"{backend}: split programs {Math.Max(d.RelL2, d.RelMax):E3} from one");
        }
    }

    /// <summary>The managed executor's attention in pieces of query rows (its score budget) gives the rows of one piece.</summary>
    [Fact]
    public void AttentionInPiecesGivesTheRowsOfOnePiece()
    {
        using var lane = new Lane(BackendType.Cpu);
        using MuseGlimmerVisionTower whole = Tower(lane, MmprojF32);
        using MuseGlimmerVisionTower pieces = Tower(lane, MmprojF32);
        pieces.AttentionChunkRows = 5;
        var (w, h, px) = Image((8, 8), 31);
        var d = Distance(whole.EncodeToHost(px, w, h, out _), pieces.EncodeToHost(px, w, h, out _));
        output.WriteLine($"pieces of 5 rows vs one: relL2 {d.RelL2:E2}, max {d.RelMax:E2}");
        Assert.True(Math.Max(d.RelL2, d.RelMax) <= 1e-6);
    }

    /// <summary>The window order, windows and merge gather of a clipped grid, by hand: 6 x 10 patches in windows of 4.</summary>
    [Fact]
    public void WindowsGroupThePatchesOfEachWindowTogether()
    {
        int[] order = MuseGlimmerVisionGrid.WindowOrder(6, 10, 4, out int[] windows);
        Assert.Equal(new[] { 0, 16, 24, 40, 48, 56, 60 }, windows);
        Assert.Equal(new[] { 0, 1, 2, 3, 6, 7, 8, 9 }, order.Take(8));
        Assert.Equal(new[] { 4, 5, 10, 11 }, order.Skip(16).Take(4));
        Assert.Equal(Enumerable.Range(0, 60), order.OrderBy(x => x));
        int[] gather = MuseGlimmerVisionGrid.MergeGather(6, 10, 2, order);
        // Block (0, 0) is raster patches 0, 1, 6, 7: window-order rows 0, 1, 4, 5.
        Assert.Equal(new[] { 0, 1, 4, 5 }, gather.Take(4));
    }
}

