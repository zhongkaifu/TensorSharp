// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// CpuFloatBench: throughput of the pure-C# CPU float kernels.
//
//   CpuFloatBench [all|peak|gemm|skinny|batch|direct|attn|eltwise|kernel|probe|narrow|attnlong]... [--quick] [--no-check]
//   CpuFloatBench shape M N K NN|NT|TN|TT        one Ops.Addmm shape (blocking sweeps)
//   CpuFloatBench qwen-te <text-encoder.gguf> [dump.bin] [compare.bin]
//                                                 Qwen-Image text encoder on the cpu backend,
//                                                 timing + parity against a previous dump
//   CpuFloatBench gemma4-vision <mmproj.gguf> <image> [dump.bin] [compare.bin]
//                                                 Gemma4 vision tower on a CpuAllocator, exactly
//                                                 as the direct `cuda` backend runs it
//
// 'kernel' times the raw microkernels on L1-resident panels; 'probe' traces per-call latency of
// a small transcendental op and the CpuParallel fork/join (warm-up and dispatch diagnostics).
// 'narrow' sweeps N for tall dot-layout products (packed tile vs narrow dot path, per kernel:
// the data behind the TS_CPU_SGEMM_DOT_MAXN defaults); 'attnlong' sweeps the CPU attention's
// query-block floor at long sequences. Neither is part of 'all'.
//
// The implementation is picked by environment variables at process start, so an A/B is two
// runs of the same binary:
//   TS_CPU_POOL=0               Core kernels fork on Parallel.For instead of CpuWorkerPool
//   TS_CPU_DISABLE_AVX512=1     AVX2 microkernels / 256-bit vectors on an AVX-512 host
// GEMM rows report GFLOPS and the max scale-relative error of 256 sampled outputs against
// a double-precision reference; elementwise rows report microseconds and GB/s moved.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;

internal static unsafe class Program
{
    private static readonly IAllocator Alloc = new CpuAllocator(BlasEnum.DotNet);
    private static bool Quick;
    private static bool Check = true;

    private static int Main(string[] args)
    {
        // Bind Core's CpuParallel to the Models worker pool exactly as a model load would.
        RuntimeHelpers.RunModuleConstructor(typeof(TensorSharp.Models.ModelBase).Module.ModuleHandle);

        var sections = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];
            if (a == "--quick") Quick = true;
            else if (a == "--no-check") Check = false;
            else if (a == "qwen-te" && i + 1 < args.Length)
            {
                // qwen-te <Qwen3VL text-encoder.gguf> [dump.bin] [compare.bin]: end-to-end encode on
                // the cpu backend (per-op trunk: batched attention GEMMs, softmax, norms).
                string te = args[i + 1];
                string dump = i + 2 < args.Length && !args[i + 2].StartsWith("-") ? args[i + 2] : null;
                string cmp = dump != null && i + 3 < args.Length && !args[i + 3].StartsWith("-") ? args[i + 3] : null;
                QwenTextEncoder(te, dump, cmp);
                i += 1 + (dump != null ? 1 : 0) + (cmp != null ? 1 : 0);
                sections.Add("none");
            }
            else if (a == "gemma4-vision" && i + 2 < args.Length)
            {
                // gemma4-vision <mmproj.gguf> <image> [dump.bin] [compare.bin]: the tower the `cuda`
                // backend keeps on a CpuAllocator (Gemma4RuntimeModel.LoadVisionEncoder), i.e. Ops.* on
                // CpuStorage - old vs new Core kernels are two runs with the TS_CPU_* switches.
                string mm = args[i + 1], img = args[i + 2];
                string dump = i + 3 < args.Length && !args[i + 3].StartsWith("-") ? args[i + 3] : null;
                string cmp = dump != null && i + 4 < args.Length && !args[i + 4].StartsWith("-") ? args[i + 4] : null;
                Gemma4Vision(mm, img, dump, cmp);
                i += 2 + (dump != null ? 1 : 0) + (cmp != null ? 1 : 0);
                sections.Add("none");
            }
            else if (a == "shape" && i + 4 < args.Length)
            {
                // shape M N K NN|NT|TN|TT : one Ops.Addmm shape (for blocking sweeps).
                int m = int.Parse(args[i + 1]), n = int.Parse(args[i + 2]), k = int.Parse(args[i + 3]);
                string o = args[i + 4].ToUpperInvariant();
                RunGemm("shape", m, n, k, o[0] == 'T', o[1] == 'T');
                i += 4;
                sections.Add("none");
            }
            else sections.Add(a.ToLowerInvariant());
        }
        if (sections.Count == 0) sections.Add("all");
        bool all = sections.Contains("all");
        if (sections.Contains("none") && sections.Count == sections.FindAll(s => s == "none").Count) return 0;

        Console.WriteLine($"CpuFloatBench  cores={Environment.ProcessorCount}  parallel={(CpuParallel.HasCustomRunner ? "CpuWorkerPool" : "Parallel.For")}x{CpuParallel.DegreeOfParallelism}");
        Console.WriteLine($"  sgemm={CpuSgemm.ActiveKernelName}  simd-elementwise={(CpuKernels.Use512 ? "on(512)" : "on(256)")}  " +
                          $"avx512f={Avx512F.IsSupported} v512={Vector512.IsHardwareAccelerated} fma={Fma.IsSupported}");
        Console.WriteLine($"  env: TS_CPU_POOL={Env("TS_CPU_POOL")} TS_CPU_DISABLE_AVX512={Env("TS_CPU_DISABLE_AVX512")}");

        if (all || sections.Contains("peak")) Peak();
        if (sections.Contains("kernel")) KernelSection();
        if (sections.Contains("probe"))
        {
            // Per-call latency trace of one small transcendental op (diagnoses warm-up/dispatch effects).
            using Tensor x = Random(12, 70, 2816);
            using Tensor r = new Tensor(Alloc, DType.Float32, 70, 2816);
            for (int pass = 0; pass < 3; pass++)
            {
                var lat = new List<double>();
                for (int i = 0; i < 300; i++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    Ops.Exp(r, x);
                    lat.Add(Stopwatch.GetElapsedTime(t0).TotalMicroseconds);
                }
                Console.WriteLine($"  exp [70,2816] pass {pass}: first {lat[0]:F0} us, min {lat.Min():F1} us, median {lat.OrderBy(v => v).ElementAt(150):F1} us, last {lat[^1]:F1} us");
            }
            foreach (int blocks in new[] { 2, 8, 25, 64 })
            {
                var lat = new List<double>();
                long sink = 0;
                for (int i = 0; i < 2000; i++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    CpuParallel.For(blocks, b => Interlocked.Increment(ref sink));
                    lat.Add(Stopwatch.GetElapsedTime(t0).TotalMicroseconds);
                }
                lat.Sort();
                Console.WriteLine($"  CpuParallel.For({blocks,2}, empty): min {lat[0]:F1} us  median {lat[1000]:F1} us  p90 {lat[1800]:F1} us");
            }
        }
        if (all || sections.Contains("gemm")) GemmSection();
        if (all || sections.Contains("skinny")) SkinnySection();
        if (all || sections.Contains("batch")) BatchSection();
        if (all || sections.Contains("direct")) DirectSection();
        if (all || sections.Contains("attn")) AttentionSection();
        if (sections.Contains("narrow")) NarrowSection();
        if (sections.Contains("attnlong")) LongAttentionSection();
        if (all || sections.Contains("eltwise")) EltwiseSection();
        return 0;
    }

    private static string Env(string name) => Environment.GetEnvironmentVariable(name) ?? "-";

    // ------------------------------------------------------------------------------------
    //  Timing
    // ------------------------------------------------------------------------------------

    private static (double best, double median) Time(Action run, double budgetSeconds = 0.6, int maxReps = 25)
    {
        // Warm-up past tiered compilation: BCL/package code (TensorPrimitives) starts at Tier0 and
        // only tiers up after ~30 calls plus a background compile, which would otherwise be what
        // the first shapes measure. Also pays the JIT, pack buffers and page faults.
        var warm = Stopwatch.StartNew();
        for (int w = 0; w < 40 && (w == 0 || warm.Elapsed.TotalSeconds < 1.5); w++) run();
        Thread.Sleep(150);
        run();
        var samples = new List<double>();
        var total = Stopwatch.StartNew();
        do
        {
            long t0 = Stopwatch.GetTimestamp();
            run();
            samples.Add(Stopwatch.GetElapsedTime(t0).TotalSeconds);
        } while (samples.Count < 3 || (total.Elapsed.TotalSeconds < (Quick ? budgetSeconds / 3 : budgetSeconds) && samples.Count < maxReps));
        samples.Sort();
        return (samples[0], samples[samples.Count / 2]);
    }

    // ------------------------------------------------------------------------------------
    //  Peak: independent FMA chains per thread (no memory traffic).
    // ------------------------------------------------------------------------------------

    private static double PeakGflops = double.NaN;

    /// <summary>Quick all-thread FMA peak (max of zmm/ymm) right before a section: the laptop's
    /// clocks drift with temperature, so %-of-peak is only meaningful against a fresh number.</summary>
    private static void MeasurePeakQuietly()
    {
        int threads = CpuParallel.DegreeOfParallelism;
        const long iters = 10_000_000;
        double best = 0;
        if (Avx512F.IsSupported)
            best = Math.Max(best, iters * 12.0 * 16 * 2 * threads / Time(() => CpuParallel.For(threads, t => Fma512(iters)), 0.3, 3).best / 1e9);
        if (Fma.IsSupported)
            best = Math.Max(best, iters * 12.0 * 8 * 2 * threads / Time(() => CpuParallel.For(threads, t => Fma256(iters)), 0.3, 3).best / 1e9);
        PeakGflops = best > 0 ? best : double.NaN;
        Console.WriteLine($"  (all-thread FMA peak now: {PeakGflops:F0} GFLOPS)");
    }

    private static void Peak()
    {
        Console.WriteLine();
        Console.WriteLine("== peak (synthetic FMA, 12 independent accumulators per thread) ==");
        int threads = CpuParallel.DegreeOfParallelism;
        foreach (bool wide in new[] { true, false })
        {
            if (wide && !Avx512F.IsSupported) continue;
            if (!wide && !Fma.IsSupported) continue;
            const long iters = 20_000_000;
            double flopPerThread = iters * 12.0 * (wide ? 16 : 8) * 2;
            var one = Time(() => { _ = wide ? Fma512(iters) : Fma256(iters); }, 0.5, 5);
            var many = Time(() => CpuParallel.For(threads, t => { _ = wide ? Fma512(iters) : Fma256(iters); }), 0.5, 5);
            Console.WriteLine($"  {(wide ? "zmm" : "ymm")}: 1 thread {flopPerThread / one.best / 1e9,7:F1} GFLOPS   {threads} threads {flopPerThread * threads / many.best / 1e9,7:F1} GFLOPS");
        }
    }

    /// <summary>Raw microkernel throughput on L1-resident packed panels (no packing, no C traffic to speak of).</summary>
    private static void KernelSection()
    {
        Console.WriteLine();
        Console.WriteLine("== microkernels on L1-resident panels (kc=256, C accumulate) ==");
        int threads = CpuParallel.DegreeOfParallelism;
        var kernels = new List<(string name, int mr, int nr, int id)>();
        if (Avx512F.IsSupported) kernels.Add(("zmm 8x32", 8, 32, 0));
        if (Avx512F.VL.IsSupported) kernels.Add(("ymm 8x24 (evex)", 8, 24, 1));
        if (Fma.IsSupported) kernels.Add(("ymm 6x16", 6, 16, 2));
        foreach (var (name, mr, nr, id) in kernels)
        {
            delegate*<int, float*, float*, float*, long, float, float, void> fn =
                id == 0 ? &CpuSgemm.Kernel8x32 : id == 1 ? &CpuSgemm.Kernel8x24 : &CpuSgemm.Kernel6x16;
            const int kc = 256, calls = 40_000;
            double flop = 2.0 * mr * nr * kc * calls;
            void Run()
            {
                float* pa = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(mr * kc * 4), 64);
                float* pb = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(nr * kc * 4), 64);
                float* c = (float*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)(mr * nr * 4), 64);
                for (int i = 0; i < mr * kc; i++) pa[i] = 1e-3f;
                for (int i = 0; i < nr * kc; i++) pb[i] = 1e-3f;
                for (int i = 0; i < mr * nr; i++) c[i] = 0;
                for (int i = 0; i < calls; i++) fn(kc, pa, pb, c, nr, 1f, 1f);
                System.Runtime.InteropServices.NativeMemory.AlignedFree(pa);
                System.Runtime.InteropServices.NativeMemory.AlignedFree(pb);
                System.Runtime.InteropServices.NativeMemory.AlignedFree(c);
            }
            var one = Time(Run, 0.5, 5);
            var many = Time(() => CpuParallel.For(threads, t => Run()), 0.5, 5);
            Console.WriteLine($"  {name,-16} 1 thread {flop / one.best / 1e9,7:F1} GFLOPS   {threads} threads {flop * threads / many.best / 1e9,7:F1} GFLOPS");
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static float Fma512(long iters)
    {
        var x = Vector512.Create(1.0000001f);
        var y = Vector512.Create(0.9999999f);
        Vector512<float> a0 = x, a1 = x, a2 = x, a3 = x, a4 = x, a5 = x, a6 = x, a7 = x, a8 = x, a9 = x, a10 = x, a11 = x;
        for (long i = 0; i < iters; i++)
        {
            a0 = Avx512F.FusedMultiplyAdd(a0, x, y); a1 = Avx512F.FusedMultiplyAdd(a1, x, y);
            a2 = Avx512F.FusedMultiplyAdd(a2, x, y); a3 = Avx512F.FusedMultiplyAdd(a3, x, y);
            a4 = Avx512F.FusedMultiplyAdd(a4, x, y); a5 = Avx512F.FusedMultiplyAdd(a5, x, y);
            a6 = Avx512F.FusedMultiplyAdd(a6, x, y); a7 = Avx512F.FusedMultiplyAdd(a7, x, y);
            a8 = Avx512F.FusedMultiplyAdd(a8, x, y); a9 = Avx512F.FusedMultiplyAdd(a9, x, y);
            a10 = Avx512F.FusedMultiplyAdd(a10, x, y); a11 = Avx512F.FusedMultiplyAdd(a11, x, y);
        }
        return Vector512.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static float Fma256(long iters)
    {
        var x = Vector256.Create(1.0000001f);
        var y = Vector256.Create(0.9999999f);
        Vector256<float> a0 = x, a1 = x, a2 = x, a3 = x, a4 = x, a5 = x, a6 = x, a7 = x, a8 = x, a9 = x, a10 = x, a11 = x;
        for (long i = 0; i < iters; i++)
        {
            a0 = Fma.MultiplyAdd(a0, x, y); a1 = Fma.MultiplyAdd(a1, x, y);
            a2 = Fma.MultiplyAdd(a2, x, y); a3 = Fma.MultiplyAdd(a3, x, y);
            a4 = Fma.MultiplyAdd(a4, x, y); a5 = Fma.MultiplyAdd(a5, x, y);
            a6 = Fma.MultiplyAdd(a6, x, y); a7 = Fma.MultiplyAdd(a7, x, y);
            a8 = Fma.MultiplyAdd(a8, x, y); a9 = Fma.MultiplyAdd(a9, x, y);
            a10 = Fma.MultiplyAdd(a10, x, y); a11 = Fma.MultiplyAdd(a11, x, y);
        }
        return Vector256.Sum(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 + a10 + a11);
    }

    // ------------------------------------------------------------------------------------
    //  GEMM through Ops.Addmm
    // ------------------------------------------------------------------------------------

    private static Tensor Random(long seed, params long[] sizes)
    {
        var t = new Tensor(Alloc, DType.Float32, sizes);
        long n = t.ElementCount();
        float* p = (float*)CpuNativeHelpers.GetBufferStart(t);
        ulong s = (ulong)seed * 0x9E3779B97F4A7C15UL + 1;
        for (long i = 0; i < n; i++)
        {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            p[i] = (float)((s >> 40) * (1.0 / (1UL << 24)) * 2.0 - 1.0);
        }
        return t;
    }

    /// <summary>A(i,p) through the view's strides.</summary>
    private static float At(Tensor t, long i, long j)
    {
        float* p = (float*)CpuNativeHelpers.GetBufferStart(t);
        return p[i * t.Strides[0] + j * t.Strides[1]];
    }

    /// <summary>Max |c - ref| / sum|a||b| over 256 sampled outputs (scale-relative error).</summary>
    private static double SampledError(Tensor a, Tensor b, Tensor c, float alpha)
    {
        if (!Check) return double.NaN;
        long m = c.Sizes[0], n = c.Sizes[1], k = a.Sizes[1];
        var rng = new System.Random(7);
        double worst = 0;
        for (int s = 0; s < 256; s++)
        {
            long i = rng.NextInt64(m), j = rng.NextInt64(n);
            double acc = 0, mag = 0;
            for (long p = 0; p < k; p++)
            {
                double x = At(a, i, p), y = At(b, p, j);
                acc += x * y;
                mag += Math.Abs(x * y);
            }
            double err = Math.Abs(At(c, i, j) - alpha * acc) / (Math.Abs(alpha) * mag + 1e-30);
            worst = Math.Max(worst, err);
        }
        return worst;
    }

    private static void RunGemm(string label, int m, int n, int k, bool transA, bool transB)
    {
        using Tensor aStore = transA ? Random(1, k, m) : Random(1, m, k);
        using Tensor bStore = transB ? Random(2, n, k) : Random(2, k, n);
        using Tensor a = transA ? aStore.Transpose() : aStore.CopyRef();
        using Tensor b = transB ? bStore.Transpose() : bStore.CopyRef();
        using Tensor c = new Tensor(Alloc, DType.Float32, m, n);
        var (best, median) = Time(() => Ops.Addmm(c, 0f, c, 1f, a, b));
        double flop = 2.0 * m * n * k;
        double err = SampledError(a, b, c, 1f);
        double pct = 100.0 * flop / best / 1e9 / PeakGflops;
        Console.WriteLine($"  {label,-26} {m,6}x{n,-6}x{k,-5} {(transA ? "T" : "N")}{(transB ? "T" : "N")}  {flop / best / 1e9,8:F1} GFLOPS {pct,4:F0}%pk (median {flop / median / 1e9,7:F1})  {best * 1e3,9:F3} ms  err {err:E1}");
    }

    private static void GemmSection()
    {
        Console.WriteLine();
        Console.WriteLine("== Ops.Addmm (F32 GEMM) ==");
        MeasurePeakQuietly();
        int[] squares = Quick ? new[] { 256, 1024 } : new[] { 128, 256, 512, 1024, 2048 };
        foreach (int s in squares)
        {
            RunGemm("square", s, s, s, false, false);
            RunGemm("square", s, s, s, false, true);
        }
        if (!Quick)
        {
            RunGemm("square", 1024, 1024, 1024, true, false);
            RunGemm("square", 4096, 4096, 4096, false, false);
            RunGemm("odd", 1000, 999, 1001, false, true);
            RunGemm("odd", 257, 131, 77, false, false);
        }
        RunGemm("attn QK^T", 1024, 1024, 128, false, true);
        RunGemm("attn PV", 1024, 128, 1024, false, false);
        RunGemm("dense [256,2816]x2112", 256, 2112, 2816, false, true);
    }

    private static void SkinnySection()
    {
        Console.WriteLine();
        Console.WriteLine("== Ops.Addmm skinny (M = 1..70) ==");
        MeasurePeakQuietly();
        int[] ms = Quick ? new[] { 1, 16, 70 } : new[] { 1, 2, 4, 8, 16, 32, 70 };
        foreach (int m in ms)
        {
            RunGemm("skinny", m, 2816, 2816, false, false);
            RunGemm("skinny", m, 2816, 2816, false, true);
            if (!Quick)
            {
                RunGemm("skinny", m, 256, 64, false, true);
                RunGemm("skinny", m, 4096, 4096, false, true);
            }
        }
    }

    // ------------------------------------------------------------------------------------
    //  Batched GEMM (attention-shaped) through Ops.AddmmBatch
    // ------------------------------------------------------------------------------------

    private static void BatchSection()
    {
        Console.WriteLine();
        Console.WriteLine("== Ops.AddmmBatch (attention: scores = Q K^T, out = P V) ==");
        foreach (var (heads, seq, hd) in new[] { (32, 23, 128), (32, 300, 128), (16, 70, 256), (24, 1024, 128) })
        {
            using Tensor q = Random(3, heads, seq, hd);
            using Tensor kS = Random(4, heads, seq, hd);
            using Tensor kT = kS.Transpose(1, 2);
            using Tensor scores = new Tensor(Alloc, DType.Float32, heads, seq, seq);
            var t1 = Time(() => Ops.AddmmBatch(scores, 0f, scores, 1f, q, kT));
            using Tensor v = Random(5, heads, seq, hd);
            using Tensor o = new Tensor(Alloc, DType.Float32, heads, seq, hd);
            var t2 = Time(() => Ops.AddmmBatch(o, 0f, o, 1f, scores, v));
            double flop = 2.0 * heads * seq * seq * hd;
            Console.WriteLine($"  heads={heads,2} seq={seq,5} hd={hd,3}   QK^T {flop / t1.best / 1e9,7:F1} GFLOPS {t1.best * 1e3,8:F3} ms   PV {flop / t2.best / 1e9,7:F1} GFLOPS {t2.best * 1e3,8:F3} ms");
        }
    }

    // ------------------------------------------------------------------------------------
    //  DirectOps CPU GEMMs (VAE im2col convs, direct linears)
    // ------------------------------------------------------------------------------------

    private static void DirectSection()
    {
        Console.WriteLine();
        Console.WriteLine("== DirectOps.CpuGemmABt / CpuGemmAB (im2col conv shapes) ==");
        MeasurePeakQuietly();
        var shapes = Quick
            ? new[] { (16384, 384, 3456), (65536, 96, 864) }
            : new[] { (16384, 384, 3456), (65536, 192, 1728), (262144, 96, 864), (4096, 384, 384), (1024, 12, 384),
                      (65536, 3, 2592), (65536, 16, 1152), (65536, 40, 1152) };
        foreach (var (m, n, k) in shapes)
        {
            using Tensor x = Random(6, m, k);
            using Tensor w = Random(7, n, k);
            using Tensor c = new Tensor(Alloc, DType.Float32, m, n);
            var t1 = Time(() => DirectOps.CpuGemmABt(x, w, c, 1f, 0f), 1.0, 8);
            using Tensor wT = w.Transpose();
            double err = SampledError(x, wT, c, 1f);
            using Tensor wKn = Random(8, k, n);
            var t2 = Time(() => DirectOps.CpuGemmAB(x, wKn, c, 1f, 0f), 1.0, 8);
            double flop = 2.0 * m * n * k;
            Console.WriteLine($"  {m,7}x{n,-4}x{k,-5}  ABt {flop / t1.best / 1e9,7:F1} GFLOPS {t1.best * 1e3,9:F2} ms (err {err:E1})   AB {flop / t2.best / 1e9,7:F1} GFLOPS {t2.best * 1e3,9:F2} ms");
        }
    }

    // ------------------------------------------------------------------------------------
    //  DirectOps.Attention CPU path
    // ------------------------------------------------------------------------------------

    private static void AttentionSection()
    {
        Console.WriteLine();
        Console.WriteLine("== DirectOps.Attention (CPU generic path) ==");
        using var ctx = new DirectContext(Alloc);
        foreach (var (heads, seq, hd) in Quick ? new[] { (24, 1024, 128) } : new[] { (24, 1024, 128), (1, 4096, 384), (16, 280, 128), (12, 2048, 64) })
        {
            using Tensor q = Random(9, seq, (long)heads * hd);
            using Tensor k = Random(10, seq, (long)heads * hd);
            using Tensor v = Random(11, seq, (long)heads * hd);
            float scale = 1f / MathF.Sqrt(hd);
            Tensor last = null;
            var t = Time(() => { last?.Dispose(); last = DirectOps.Attention(ctx, q, k, v, heads, hd, scale); }, 1.0, 10);
            double err = Check ? AttentionError(q, k, v, last, heads, seq, hd, scale) : double.NaN;
            last?.Dispose();
            double flop = 4.0 * heads * seq * seq * hd;
            Console.WriteLine($"  heads={heads,2} seq={seq,5} hd={hd,3}  {flop / t.best / 1e9,7:F1} GFLOPS {t.best * 1e3,9:F2} ms  (max abs err vs fp64 on 16 rows {err:E1})");
        }
    }

    private static double AttentionError(Tensor q, Tensor k, Tensor v, Tensor o, int heads, int seq, int hd, float scale)
    {
        float* qp = (float*)CpuNativeHelpers.GetBufferStart(q);
        float* kp = (float*)CpuNativeHelpers.GetBufferStart(k);
        float* vp = (float*)CpuNativeHelpers.GetBufferStart(v);
        float* op = (float*)CpuNativeHelpers.GetBufferStart(o);
        long row = (long)heads * hd;
        var rng = new System.Random(3);
        double worst = 0;
        var s = new double[seq];
        for (int t = 0; t < 16; t++)
        {
            int h = rng.Next(heads), i = rng.Next(seq);
            double max = double.NegativeInfinity;
            for (int j = 0; j < seq; j++)
            {
                double d = 0;
                for (int e = 0; e < hd; e++) d += (double)qp[i * row + h * hd + e] * kp[j * row + h * hd + e];
                s[j] = d * scale;
                max = Math.Max(max, s[j]);
            }
            double sum = 0;
            for (int j = 0; j < seq; j++) { s[j] = Math.Exp(s[j] - max); sum += s[j]; }
            for (int e = 0; e < hd; e++)
            {
                double acc = 0;
                for (int j = 0; j < seq; j++) acc += s[j] / sum * vp[j * row + h * hd + e];
                worst = Math.Max(worst, Math.Abs(acc - op[i * row + h * hd + e]));
            }
        }
        return worst;
    }

    // ------------------------------------------------------------------------------------
    //  Narrow N sweep: packed register tile vs the narrow dot path, per kernel
    // ------------------------------------------------------------------------------------

    /// <summary>Best of a few runs after a short warm-up (the GEMM code is always fully JIT-optimized).</summary>
    private static double TimeBest(Action run, int reps)
    {
        run();
        run();
        double best = double.MaxValue;
        for (int r = 0; r < reps; r++)
        {
            long t0 = Stopwatch.GetTimestamp();
            run();
            best = Math.Min(best, Stopwatch.GetElapsedTime(t0).TotalSeconds);
        }
        return best;
    }

    private static void NarrowSection()
    {
        Console.WriteLine();
        Console.WriteLine("== narrow N (DirectOps.CpuGemmABt: A[M,K] x B[N,K]^T): packed tile vs dot path ==");
        MeasurePeakQuietly();
        CpuSgemm.KernelKind saved = CpuSgemm.ActiveKernel;
        var kinds = Quick
            ? new[] { saved }
            : Enum.GetValues<CpuSgemm.KernelKind>().Where(CpuSgemm.IsSupported).Reverse().ToArray();
        int[] ns = Quick ? new[] { 3, 16, 32, 48, 64 } : new[] { 1, 3, 8, 12, 16, 24, 32, 40, 48, 64, 96 };
        var shapes = Quick ? new[] { (65536, 1152) } : new[] { (65536, 1152), (65536, 288), (8192, 4096), (300, 2816) };
        try
        {
            foreach (CpuSgemm.KernelKind kind in kinds)
            {
                CpuSgemm.ActiveKernel = kind;
                Console.WriteLine($"  kernel {CpuSgemm.ActiveKernelName} (default dot max N {DefaultDotMaxN()})");
                foreach (var (m, k) in shapes)
                {
                    using Tensor x = Random(31, m, k);
                    foreach (int n in ns)
                    {
                        using Tensor w = Random(32, n, k);
                        using Tensor c = new Tensor(Alloc, DType.Float32, m, n);
                        double flop = 2.0 * m * n * k;
                        int reps = flop > 2e9 ? 3 : 7;
                        double tPacked = double.MaxValue, tDot = double.MaxValue;
                        for (int round = 0; round < 2; round++)   // interleaved A/B
                        {
                            CpuSgemm.NarrowDotMaxN = 0;
                            tPacked = Math.Min(tPacked, TimeBest(() => DirectOps.CpuGemmABt(x, w, c, 1f, 0f), reps));
                            CpuSgemm.NarrowDotMaxN = int.MaxValue;
                            tDot = Math.Min(tDot, TimeBest(() => DirectOps.CpuGemmABt(x, w, c, 1f, 0f), reps));
                        }
                        using Tensor wT = w.Transpose();
                        double err = SampledError(x, wT, c, 1f);
                        Console.WriteLine($"    {m,6}x{n,-3}x{k,-5} packed {flop / tPacked / 1e9,7:F1} GFLOPS {tPacked * 1e3,8:F2} ms   dot {flop / tDot / 1e9,7:F1} GFLOPS {tDot * 1e3,8:F2} ms   dot/packed {tPacked / tDot,5:F2}x  (dot err {err:E1})");
                    }
                }
            }
        }
        finally
        {
            CpuSgemm.ActiveKernel = saved;
            CpuSgemm.NarrowDotMaxN = -1;
        }
    }

    private static int DefaultDotMaxN()
    {
        CpuSgemm.NarrowDotMaxN = -1;
        return CpuSgemm.NarrowDotMaxN;
    }

    /// <summary>CPU attention at long sequences (DiT/VAE image tokens) across query-block floors:
    /// every block re-packs its head's K^T and V panels, so fewer, taller blocks amortize that.</summary>
    private static void LongAttentionSection()
    {
        Console.WriteLine();
        Console.WriteLine("== DirectOps.Attention, long sequences: query-block floor sweep ==");
        using var ctx = new DirectContext(Alloc);
        int savedFloor = DirectOps.CpuAttentionMinQBlock;
        int[] floors = Quick ? new[] { 16, 64 } : new[] { 16, 32, 64, 128 };
        var shapes = Quick ? new[] { (1, 16384, 128) } : new[] { (24, 4096, 128), (4, 8192, 128), (1, 16384, 128) };
        try
        {
            foreach (var (heads, seq, hd) in shapes)
            {
                using Tensor q = Random(41, seq, (long)heads * hd);
                using Tensor k = Random(42, seq, (long)heads * hd);
                using Tensor v = Random(43, seq, (long)heads * hd);
                float scale = 1f / MathF.Sqrt(hd);
                double flop = 4.0 * heads * seq * seq * hd;
                var best = new double[floors.Length];
                Array.Fill(best, double.MaxValue);
                for (int round = 0; round < 2; round++)   // interleaved across floors
                {
                    for (int f = 0; f < floors.Length; f++)
                    {
                        DirectOps.CpuAttentionMinQBlock = floors[f];
                        best[f] = Math.Min(best[f], TimeBest(() => DirectOps.Attention(ctx, q, k, v, heads, hd, scale).Dispose(), 2));
                    }
                }
                var cells = floors.Select((fl, f) => $"floor {fl,3}: {flop / best[f] / 1e9,6:F1} GFLOPS {best[f] * 1e3,8:F1} ms");
                Console.WriteLine($"  heads={heads,2} seq={seq,5} hd={hd,3}  " + string.Join("   ", cells));
            }
        }
        finally
        {
            DirectOps.CpuAttentionMinQBlock = savedFloor;
        }
    }


    // ------------------------------------------------------------------------------------
    //  Qwen-Image text encoder (Qwen3-VL trunk) on the cpu backend
    // ------------------------------------------------------------------------------------

    private static void QwenTextEncoder(string gguf, string dump, string compare)
    {
        Console.WriteLine($"qwen-te: {gguf}  env TS_CPU_POOL={Env("TS_CPU_POOL")}");
        var load = Stopwatch.StartNew();
        // TE_BACKEND=ggml_cpu produces the native reference dump (needs GgmlOps next to the exe).
        var backend = Env("TE_BACKEND") == "ggml_cpu" ? TensorSharp.Runtime.BackendType.GgmlCpu : TensorSharp.Runtime.BackendType.Cpu;
        Console.WriteLine($"  backend {backend}");
        using var te = new TensorSharp.Models.QwenImage.QwenImageTextEncoder(gguf, backend);
        Console.WriteLine($"  load {load.Elapsed.TotalSeconds:F1} s");
        string prompt = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n" +
                        "<|im_start|>user\nA small orange cat beside a blue ceramic vase, soft daylight, detailed photograph<|im_end|>\n<|im_start|>assistant\n";
        int[] tokens = te.Tokenizer.Encode(prompt, addSpecial: false).ToArray();
        float[] hidden = null;
        var times = new List<double>();
        for (int r = 0; r < 3; r++)
        {
            var sw = Stopwatch.StartNew();
            hidden = te.EncodeHidden(tokens);
            times.Add(sw.Elapsed.TotalMilliseconds);
            Console.WriteLine($"  encode {tokens.Length} tokens: {times[^1]:F1} ms");
        }
        Console.WriteLine($"  best {times.Min():F1} ms");
        DumpAndCompare(hidden, 0, dump, compare);
    }

    /// <summary>Writes <paramref name="values"/> to <paramref name="dump"/> and compares them with an
    /// earlier dump: max |diff|, overall cosine and (given a row width) the worst row cosine.</summary>
    private static void DumpAndCompare(float[] values, int rowWidth, string dump, string compare)
    {
        if (dump != null)
        {
            using var w = new BinaryWriter(File.Create(dump));
            w.Write(values.Length);
            foreach (float v in values) w.Write(v);
        }
        if (compare != null && File.Exists(compare))
        {
            using var rd = new BinaryReader(File.OpenRead(compare));
            int n = rd.ReadInt32();
            if (n != values.Length)
            {
                Console.WriteLine($"  vs {compare}: length {n} != {values.Length}");
                return;
            }
            double maxAbs = 0, maxRef = 0, dot = 0, na = 0, nb = 0;
            double rDot = 0, rNa = 0, rNb = 0, worstRow = 1;
            for (int i = 0; i < n; i++)
            {
                float refV = rd.ReadSingle();
                maxAbs = Math.Max(maxAbs, Math.Abs(refV - values[i]));
                maxRef = Math.Max(maxRef, Math.Abs(refV));
                dot += (double)refV * values[i]; na += (double)refV * refV; nb += (double)values[i] * values[i];
                rDot += (double)refV * values[i]; rNa += (double)refV * refV; rNb += (double)values[i] * values[i];
                if (rowWidth > 0 && (i + 1) % rowWidth == 0)
                {
                    worstRow = Math.Min(worstRow, rDot / Math.Sqrt(rNa * rNb + 1e-300));
                    rDot = rNa = rNb = 0;
                }
            }
            string rows = rowWidth > 0 ? $"  worst row cosine {worstRow:F8}" : "";
            Console.WriteLine($"  vs {compare}: max|diff| {maxAbs:E3} (max|ref| {maxRef:F2})  cosine {dot / Math.Sqrt(na * nb):F8}{rows}");
        }
    }

    // ------------------------------------------------------------------------------------
    //  Gemma4 vision tower on a CpuAllocator (the direct cuda backend's CPU half)
    // ------------------------------------------------------------------------------------

    private static void Gemma4Vision(string mmproj, string image, string dump, string compare)
    {
        Console.WriteLine($"gemma4-vision: {mmproj}  env TS_CPU_POOL={Env("TS_CPU_POOL")} TS_CPU_DISABLE_AVX512={Env("TS_CPU_DISABLE_AVX512")}");
        var load = Stopwatch.StartNew();
        using var enc = new TensorSharp.Models.Gemma4VisionEncoder(mmproj, new CpuAllocator(BlasEnum.DotNet));
        var proc = enc.IsUnified
            ? new TensorSharp.Models.Gemma4ImageProcessor(imageMean: enc.ImageMean, imageStd: enc.ImageStd)
            : new TensorSharp.Models.Gemma4ImageProcessor();
        var (pixels, w, h) = proc.ProcessImage(image);
        Console.WriteLine($"  load {load.Elapsed.TotalSeconds:F1} s, image {w}x{h}");
        float[] output = null;
        int rowWidth = 0;
        var times = new List<double>();
        for (int r = 0; r < 3; r++)
        {
            var sw = Stopwatch.StartNew();
            using Tensor o = enc.Encode(pixels, w, h);
            using Tensor contig = Ops.NewContiguous(o);
            output = contig.GetElementsAsFloat((int)contig.ElementCount());
            times.Add(sw.Elapsed.TotalMilliseconds);
            rowWidth = (int)contig.Sizes[^1];
            Console.WriteLine($"  encode -> [{string.Join(",", contig.Sizes.ToArray())}]: {times[^1]:F1} ms");
        }
        Console.WriteLine($"  best {times.Min():F1} ms");
        DumpAndCompare(output, rowWidth, dump, compare);
    }

    // ------------------------------------------------------------------------------------
    //  Elementwise / norm / softmax / copy
    // ------------------------------------------------------------------------------------

    private static void Row(string op, string shape, double seconds, double bytes)
        => Console.WriteLine($"  {op,-18} {shape,-16} {seconds * 1e6,11:F1} us  {bytes / seconds / 1e9,7:F1} GB/s");

    private static void EltwiseSection()
    {
        Console.WriteLine();
        Console.WriteLine("== elementwise / norm / softmax / copy (best of N) ==");
        var shapes = Quick
            ? new[] { (70L, 2816L), (256L, 2816L), (4096L, 4096L) }
            : new[] { (70L, 2816L), (256L, 2816L), (300L, 2112L), (4096L, 4096L), (4096L, 12288L) };
        foreach (var (rows, cols) in shapes)
        {
            string shape = $"[{rows},{cols}]";
            double n = rows * cols;
            using Tensor x = Random(12, rows, cols);
            using Tensor y = Random(13, rows, cols);
            using Tensor r = new Tensor(Alloc, DType.Float32, rows, cols);
            using Tensor g = Random(14, cols);
            using Tensor bias = Random(15, cols);

            Row("add", shape, Time(() => Ops.Add(r, x, y)).best, 12 * n);
            Row("add (row bcast)", shape, Time(() => Ops.Add(r, x, bias)).best, 8 * n);
            Row("mul scalar", shape, Time(() => Ops.Mul(r, x, 1.7f)).best, 8 * n);
            Row("rmsnorm", shape, Time(() => Ops.RMSNorm(r, x, g, null, 1e-6f)).best, 8 * n);
            Row("layernorm", shape, Time(() => Ops.LayerNorm(r, x, g, bias, 1e-6f)).best, 8 * n);
            Row("gelu", shape, Time(() => Ops.GELU(r, x)).best, 8 * n);
            Row("gelumul", shape, Time(() => Ops.GELUMul(r, x, y)).best, 12 * n);
            Row("silu", shape, Time(() => Ops.SiLU(r, x)).best, 8 * n);
            Row("silumul", shape, Time(() => Ops.SiLUMul(r, x, y)).best, 12 * n);
            Row("sigmoid", shape, Time(() => Ops.Sigmoid(r, x)).best, 8 * n);
            Row("tanh", shape, Time(() => Ops.Tanh(r, x)).best, 8 * n);
            Row("exp", shape, Time(() => Ops.Exp(r, x)).best, 8 * n);
            Row("softmax", shape, Time(() => Ops.Softmax(r, x)).best, 8 * n);
            using (Tensor narrowSrc = x.Narrow(1, 0, cols / 2))
            using (Tensor half = new Tensor(Alloc, DType.Float32, rows, cols / 2))
            {
                Row("copy narrow->contig", shape, Time(() => Ops.Copy(half, narrowSrc)).best, 4 * n);
            }
        }

        foreach (var (b, s1, s2) in new[] { (32L, 1024L, 1024L), (16L, 70L, 70L), (32L, 300L, 300L) })
        {
            using Tensor x = Random(16, b, s1, s2);
            using Tensor r = new Tensor(Alloc, DType.Float32, b, s1, s2);
            Row("softmax", $"[{b},{s1},{s2}]", Time(() => Ops.Softmax(r, x)).best, 8.0 * b * s1 * s2);
            Row("causal mask", $"[{b},{s1},{s2}]", Time(() => Ops.AddCausalMask(r, (int)s1, 0, float.NegativeInfinity)).best, 4.0 * b * s1 * s2);
        }

        // Head split/merge: [seq, heads, hd] <-> [heads, seq, hd] (permute copies).
        foreach (var (seq, heads, hd) in new[] { (70L, 16L, 256L), (1024L, 24L, 128L) })
        {
            using Tensor x = Random(17, seq, heads, hd);
            using Tensor perm = x.Transpose(0, 1);
            using Tensor r = new Tensor(Alloc, DType.Float32, heads, seq, hd);
            Row("copy permute", $"[{seq},{heads},{hd}]", Time(() => Ops.Copy(r, perm)).best, 8.0 * seq * heads * hd);
        }

        // RoPE (NeoX), rows laid out [seq, heads] with one position per row.
        foreach (var (seq, heads, hd) in new[] { (70, 16, 256), (1024, 24, 128) })
        {
            using Tensor x = Random(18, 1, seq, heads, hd);
            using Tensor pos = new Tensor(Alloc, DType.Int32, (long)seq * heads);
            int[] p = new int[seq * heads];
            for (int t = 0; t < seq; t++) for (int h = 0; h < heads; h++) p[t * heads + h] = t;
            pos.SetElementsAsInt(p);
            Row("rope_ex neox", $"[{seq},{heads},{hd}]", Time(() => Ops.RoPEEx(x, x, pos, hd, 2, 0, 10000f, 1f, 0f, 1f, 0f, 0f)).best, 8.0 * seq * heads * hd);
        }
    }
}
