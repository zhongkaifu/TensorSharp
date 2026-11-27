// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Packed, cache-blocked F32 GEMM for the pure-C# CPU backend:
    /// <c>C = alpha * op(A) * op(B) + beta * C</c> over arbitrary element strides.
    ///
    /// Layout: A(i,p) = a[i*aRowStride + p*aColStride], B(p,j) = b[p*bRowStride + j*bColStride],
    /// C(i,j) = c[i*ldc + j]. Transposes are just swapped strides, so one driver covers
    /// NN/NT/TN/TT and every tensor view Ops.Addmm hands it (row- or column-major, narrowed rows).
    ///
    /// Structure (BLIS): C is split into independent tiles run in parallel on CpuParallel; each
    /// tile loops NC (columns) -> KC (depth, B block packed into NR-wide panels) -> MC (rows, A
    /// block packed into MR-tall panels) -> an MR x NR register-tile microkernel. Every tile packs
    /// its own panels into thread-local buffers, so there are no barriers and no shared scratch;
    /// the partition keeps tiles square-ish so the re-packing that costs is ~1/tile-dim of the
    /// arithmetic. Microkernels: AVX-512 8x32 (16 zmm accumulators), AVX2+FMA 6x16 (12 ymm),
    /// and a portable Vector&lt;T&gt; 4x(2*lanes) for everything else (ARM64 included). Skinny
    /// products (M &lt;= 4) skip packing: packing B costs as much as their whole arithmetic.
    /// Narrow products in the dot layout (small N, A rows and B columns contiguous along K)
    /// skip it too: a register tile padded out to NR columns would mostly multiply zeros.
    ///
    /// Knobs: TS_CPU_DISABLE_AVX512=1 forces the AVX2 kernel (so it is testable on an AVX-512
    /// host; see <see cref="CpuIsa"/>); TS_CPU_SGEMM_KC / _MC / _NC override the cache blocking for tuning;
    /// TS_CPU_SGEMM_DOT_MAXN sets the widest N of the narrow dot path (0 turns it off).
    /// </summary>
    internal static unsafe class CpuSgemm
    {
        /// <summary>Microkernel families. Avx2Wide is the 8x24 ymm tile that needs the 32-register
        /// EVEX file, so it exists only on AVX-512 hardware.</summary>
        internal enum KernelKind { Portable = 0, Avx2 = 1, Avx2Wide = 2, Avx512 = 3 }

        private static KernelKind _kernel = DefaultKernel();

        /// <summary>The microkernel family in use (tests/benchmarks may pin another supported one).</summary>
        internal static KernelKind ActiveKernel
        {
            get => _kernel;
            set
            {
                if (!IsSupported(value))
                    throw new PlatformNotSupportedException($"{value} kernel is not supported on this CPU.");
                _kernel = value;
            }
        }

        /// <summary>Human-readable name of the active microkernel.</summary>
        public static string ActiveKernelName => _kernel switch
        {
            KernelKind.Avx512 => "avx512-8x32",
            KernelKind.Avx2Wide => "avx2-evex-8x24",
            KernelKind.Avx2 => "avx2-6x16",
            _ => $"portable-4x{2 * Vector<float>.Count}",
        };

        internal static bool IsSupported(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => CpuIsa.HasAvx512,
            // ymm-only, so it needs the EVEX register file but not an accelerated Vector512.
            KernelKind.Avx2Wide => Avx512F.VL.IsSupported && Fma.IsSupported,
            KernelKind.Avx2 => CpuIsa.HasAvx2Fma,
            _ => true,
        };

        /// <summary>
        /// TS_CPU_SGEMM_KERNEL=avx512|avx2wide|avx2|portable pins a kernel; otherwise the widest
        /// supported one, or the AVX2 6x16 tile when TS_CPU_DISABLE_AVX512=1 (the path a CPU
        /// without AVX-512 takes).
        /// </summary>
        internal static KernelKind DefaultKernel()
        {
            KernelKind? pinned = Environment.GetEnvironmentVariable("TS_CPU_SGEMM_KERNEL")?.ToLowerInvariant() switch
            {
                "avx512" => KernelKind.Avx512,
                "avx2wide" => KernelKind.Avx2Wide,
                "avx2" => KernelKind.Avx2,
                "portable" => KernelKind.Portable,
                _ => null,
            };
            if (pinned.HasValue && IsSupported(pinned.Value) &&
                !(CpuIsa.Avx512DisabledByEnv && pinned.Value >= KernelKind.Avx2Wide))
            {
                return pinned.Value;
            }

            if (CpuIsa.Avx512) return KernelKind.Avx512;
            if (CpuIsa.Avx2Fma) return KernelKind.Avx2;
            return KernelKind.Portable;
        }

        // Rows handled by the unpacked skinny path. Up to here the arithmetic per B element
        // (M FMAs) is below the cost of copying that element into a panel. The skinny path sums
        // in another order than the packed tile and the narrow dot path, so a row of a product of
        // up to this many rows rounds apart from the same row of a wider product, on every ISA
        // (tests that compare steps of different row counts read it: DEVELOPMENT.md, "Rounding by
        // row count").
        internal const int SkinnyMaxM = 4;

        // Below this much arithmetic a fork/join costs more than it saves.
        private const double ParallelFlopThreshold = 4e6;

        // Arithmetic per parallel tile we aim for at least (keeps dispatch < a few % of the tile).
        private const double MinTileFlops = 1e6;

        // TS_CPU_SGEMM_DOT_MAXN: widest N the narrow dot path takes (0 disables it). Unset, the
        // per-kernel crossover measured against the packed path is used (DefaultNarrowDotMaxN).
        private static int _dotMaxNOverride =
            int.TryParse(Environment.GetEnvironmentVariable("TS_CPU_SGEMM_DOT_MAXN"), out int dm) && dm >= 0 ? dm : -1;

        /// <summary>Widest N routed to the narrow dot path (tests/benchmarks may override it;
        /// a negative value restores the per-kernel default).</summary>
        internal static int NarrowDotMaxN
        {
            get => _dotMaxNOverride >= 0 ? _dotMaxNOverride : DefaultNarrowDotMaxN(_kernel);
            set => _dotMaxNOverride = value;
        }

        private static readonly int EnvKc = EnvInt("TS_CPU_SGEMM_KC");
        private static readonly int EnvMc = EnvInt("TS_CPU_SGEMM_MC");
        private static readonly int EnvNc = EnvInt("TS_CPU_SGEMM_NC");

        private static int EnvInt(string name)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : 0;

        private readonly struct Blocking
        {
            public readonly KernelKind Kind;
            public readonly int MR, NR, KC, MC, NC;

            public Blocking(KernelKind kind, int mr, int nr, int kc, int mc, int nc)
            {
                Kind = kind;
                MR = mr;
                NR = nr;
                KC = EnvKc > 0 ? EnvKc : kc;
                // MC/NC are kept multiples of the register tile.
                MC = RoundUp(EnvMc > 0 ? EnvMc : mc, mr);
                NC = RoundUp(EnvNc > 0 ? EnvNc : nc, nr);
            }
        }

        // KC x NR B micro-panel (32 KB at 256 x 32) stays in the 48 KB L1D across the MR
        // loop; the MC x KC A block (144 KB) stays in L2 across the NR loop; the KC x NC
        // B block (1 MB) lives in L2/L3 for the whole MC loop.
        private static Blocking GetBlocking(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => new Blocking(kind, 8, 32, 256, 144, 1024),
            KernelKind.Avx2Wide => new Blocking(kind, 8, 24, 256, 144, 1008),
            KernelKind.Avx2 => new Blocking(kind, 6, 16, 256, 144, 1024),
            _ => new Blocking(kind, 4, 2 * Vector<float>.Count, 256, 128, 1024),
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int CeilDiv(int value, int divisor) => (value + divisor - 1) / divisor;

        // ------------------------------------------------------------------------------------
        //  Public entry points
        // ------------------------------------------------------------------------------------

        /// <summary>
        /// C[m,n] = alpha * A * B + beta * C with A(i,p) = a[i*aRowStride + p*aColStride],
        /// B(p,j) = b[p*bRowStride + j*bColStride], C(i,j) = c[i*ldc + j]. beta == 0 never reads C.
        /// </summary>
        public static void Gemm(int m, int n, int k, float alpha,
            float* a, long aRowStride, long aColStride,
            float* b, long bRowStride, long bColStride,
            float beta, float* c, long ldc, bool allowParallel = true)
        {
            GemmBatched(1, m, n, k, alpha, a, 0, aRowStride, aColStride, b, 0, bRowStride, bColStride,
                beta, c, 0, ldc, allowParallel);
        }

        /// <summary>Strided-batch form of <see cref="Gemm"/> (batch offsets in elements).</summary>
        public static void GemmBatched(int batch, int m, int n, int k, float alpha,
            float* a, long aBatchStride, long aRowStride, long aColStride,
            float* b, long bBatchStride, long bRowStride, long bColStride,
            float beta, float* c, long cBatchStride, long ldc, bool allowParallel = true)
        {
            if (batch <= 0 || m <= 0 || n <= 0) return;

            if (k <= 0 || alpha == 0f)
            {
                for (int bi = 0; bi < batch; bi++)
                    ScaleC(m, n, beta, c + bi * cBatchStride, ldc);
                return;
            }

            Blocking blk = GetBlocking(_kernel);
            double itemFlops = 2.0 * m * n * k;
            double totalFlops = itemFlops * batch;
            int threads = allowParallel ? CpuParallel.DegreeOfParallelism : 1;

            if (m > SkinnyMaxM && aColStride == 1 && bRowStride == 1 && n <= NarrowDotMaxN && k >= DotLanes(_kernel))
            {
                NarrowDot(batch, m, n, k, alpha, a, aBatchStride, aRowStride, b, bBatchStride, bColStride,
                    beta, c, cBatchStride, ldc, threads, totalFlops);
                return;
            }

            bool skinny = m <= SkinnyMaxM && (bColStride == 1 || (bRowStride == 1 && aColStride == 1));

            if (threads <= 1 || totalFlops < ParallelFlopThreshold)
            {
                for (int bi = 0; bi < batch; bi++)
                {
                    RunTile(blk, skinny, m, n, k, alpha,
                        a + bi * aBatchStride, aRowStride, aColStride,
                        b + bi * bBatchStride, bRowStride, bColStride,
                        beta, c + bi * cBatchStride, ldc);
                }
                return;
            }

            // Never cut tiles so small that dispatch dominates them, nor into more blocks than
            // a few per thread (each extra tile re-packs its panels).
            int maxTotal = (int)Math.Max(1, Math.Min(8.0 * threads, totalFlops / MinTileFlops));
            ChoosePartition(blk, skinny, m, n, batch, threads, maxTotal, out int mSplits, out int nSplits,
                out int rowsPerTile, out int colsPerTile);
            int tilesPerItem = mSplits * nSplits;

            CpuParallel.For(batch * tilesPerItem, t =>
            {
                int bi = t / tilesPerItem;
                int tile = t - bi * tilesPerItem;
                int ti = tile / nSplits;
                int tj = tile - ti * nSplits;
                int r0 = ti * rowsPerTile;
                int c0 = tj * colsPerTile;
                int rows = Math.Min(rowsPerTile, m - r0);
                int cols = Math.Min(colsPerTile, n - c0);
                if (rows <= 0 || cols <= 0) return;

                RunTile(blk, skinny, rows, cols, k, alpha,
                    a + bi * aBatchStride + r0 * aRowStride, aRowStride, aColStride,
                    b + bi * bBatchStride + c0 * bColStride, bRowStride, bColStride,
                    beta, c + bi * cBatchStride + r0 * ldc + c0, ldc);
            });
        }

        /// <summary>
        /// Pick the mSplits x nSplits grid of tiles per batch item (tile edges multiples of the
        /// register tile) that minimizes estimated time. Every tile re-packs the A rows and B
        /// columns it touches, which costs about 5/tileCols + 5/tileRows of the arithmetic (a
        /// packed element takes ~0.3 cycles; one FMA lane of arithmetic ~1/16), and the job ends
        /// with its slowest thread, so the estimate is (1 + packing) / balance efficiency, where
        /// efficiency compares the ideal per-thread work with rounds x largest tile. Grids with
        /// fewer than ~2 tiles per thread are only taken when the shape allows no more, so a
        /// descheduled worker cannot stall the whole job.
        /// </summary>
        private static void ChoosePartition(in Blocking blk, bool skinny, int m, int n, int batch, int threads, int maxTotal,
            out int mSplits, out int nSplits, out int rowsPerTile, out int colsPerTile)
        {
            int rowUnit = skinny ? m : blk.MR;
            int colUnit = skinny ? Math.Max(64, 2 * Vector<float>.Count) : blk.NR;
            int mUnits = skinny ? 1 : CeilDiv(m, rowUnit);
            int nUnits = CeilDiv(n, colUnit);
            int maxPerItem = Math.Max(1, maxTotal / batch);
            int minTotal = Math.Min(maxTotal, 2 * threads);

            int bestRowsUnits = mUnits, bestColsUnits = nUnits;
            double bestScore = double.MaxValue;
            bool bestMeetsMin = false;
            for (int ms = 1; ms <= Math.Min(mUnits, maxPerItem); ms++)
            {
                int rowsUnits = CeilDiv(mUnits, ms);
                int realM = CeilDiv(mUnits, rowsUnits);
                if (realM != ms) continue; // same grid as a smaller ms
                for (int ns = 1; ns <= Math.Min(nUnits, maxPerItem / ms); ns++)
                {
                    int colsUnits = CeilDiv(nUnits, ns);
                    int realN = CeilDiv(nUnits, colsUnits);
                    if (realN != ns) continue;

                    int tiles = batch * realM * realN;
                    double tileRows = Math.Min(m, (double)rowsUnits * rowUnit);
                    double tileCols = Math.Min(n, (double)colsUnits * colUnit);
                    int rounds = CeilDiv(tiles, threads);
                    double efficiency = ((double)batch * m * n / threads) / (rounds * tileRows * tileCols);
                    double score = (1.0 + 5.0 / tileRows + 5.0 / tileCols) / Math.Min(1.0, efficiency);
                    bool meetsMin = tiles >= minTotal;
                    if ((meetsMin && !bestMeetsMin) || (meetsMin == bestMeetsMin && score < bestScore - 1e-12))
                    {
                        bestScore = score;
                        bestMeetsMin = meetsMin;
                        bestRowsUnits = rowsUnits;
                        bestColsUnits = colsUnits;
                    }
                }
            }

            rowsPerTile = skinny ? m : bestRowsUnits * rowUnit;
            colsPerTile = bestColsUnits * colUnit;
            mSplits = CeilDiv(m, rowsPerTile);
            nSplits = CeilDiv(n, colsPerTile);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void RunTile(in Blocking blk, bool skinny, int m, int n, int k, float alpha,
            float* a, long ars, long acs, float* b, long brs, long bcs, float beta, float* c, long ldc)
        {
            if (skinny)
            {
                if (bcs == 1) SkinnyRowB(m, n, k, alpha, a, ars, acs, b, brs, beta, c, ldc);
                else SkinnyDot(m, n, k, alpha, a, ars, b, bcs, beta, c, ldc);
                return;
            }

            GemmSerial(blk, m, n, k, alpha, a, ars, acs, b, brs, bcs, beta, c, ldc);
        }

        private static void ScaleC(int m, int n, float beta, float* c, long ldc)
        {
            if (beta == 1f) return;
            for (int i = 0; i < m; i++)
            {
                Span<float> row = new Span<float>(c + i * ldc, n);
                if (beta == 0f) row.Clear();
                else System.Numerics.Tensors.TensorPrimitives.Multiply(row, beta, row);
            }
        }

        // ------------------------------------------------------------------------------------
        //  Serial blocked driver (one tile)
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void GemmSerial(in Blocking blk, int m, int n, int k, float alpha,
            float* a, long ars, long acs, float* b, long brs, long bcs, float beta, float* c, long ldc)
        {
            int MR = blk.MR, NR = blk.NR;

            // Even K blocks: a 256 + 44 split would run the tail block at a fraction of peak
            // (the C tile load/store is amortized over only 44 steps).
            int kBlocks = CeilDiv(k, blk.KC);
            int KC = CeilDiv(k, kBlocks);
            int MC = Math.Min(blk.MC, RoundUp(m, MR));
            int NC = Math.Min(blk.NC, RoundUp(n, NR));

            float* pa = ThreadBuffers.GetA((long)MC * KC);
            float* pb = ThreadBuffers.GetB((long)NC * KC);
            float* edge = stackalloc float[8 * 32];

            for (int jc = 0; jc < n; jc += NC)
            {
                int nc = Math.Min(NC, n - jc);
                for (int pc = 0; pc < k; pc += KC)
                {
                    int kc = Math.Min(KC, k - pc);
                    float betaEff = pc == 0 ? beta : 1f;
                    PackB(kc, nc, b + pc * brs + jc * bcs, brs, bcs, pb, NR);

                    for (int ic = 0; ic < m; ic += MC)
                    {
                        int mc = Math.Min(MC, m - ic);
                        PackA(kc, mc, a + ic * ars + pc * acs, ars, acs, pa, MR);

                        for (int jr = 0; jr < nc; jr += NR)
                        {
                            int nr = Math.Min(NR, nc - jr);
                            float* bPanel = pb + (long)(jr / NR) * NR * kc;
                            for (int ir = 0; ir < mc; ir += MR)
                            {
                                int mr = Math.Min(MR, mc - ir);
                                float* aPanel = pa + (long)(ir / MR) * MR * kc;
                                float* cTile = c + (ic + ir) * ldc + jc + jr;
                                if (mr == MR && nr == NR)
                                {
                                    Micro(blk.Kind, kc, aPanel, bPanel, cTile, ldc, alpha, betaEff);
                                }
                                else
                                {
                                    // Edge tile: full register tile into scratch, then merge the
                                    // valid part (panels are zero padded, so the rest is 0).
                                    Micro(blk.Kind, kc, aPanel, bPanel, edge, NR, 1f, 0f);
                                    MergeEdge(edge, NR, cTile, ldc, mr, nr, alpha, betaEff);
                                }
                            }
                        }
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void MergeEdge(float* tmp, int tmpStride, float* c, long ldc, int mr, int nr, float alpha, float beta)
        {
            for (int i = 0; i < mr; i++)
            {
                float* src = tmp + i * tmpStride;
                float* dst = c + i * ldc;
                if (beta == 0f)
                {
                    for (int j = 0; j < nr; j++) dst[j] = alpha * src[j];
                }
                else if (beta == 1f)
                {
                    for (int j = 0; j < nr; j++) dst[j] += alpha * src[j];
                }
                else
                {
                    for (int j = 0; j < nr; j++) dst[j] = alpha * src[j] + beta * dst[j];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Micro(KernelKind kind, int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            if (kind == KernelKind.Avx512) Kernel8x32(kc, pa, pb, c, ldc, alpha, beta);
            else if (kind == KernelKind.Avx2Wide) Kernel8x24(kc, pa, pb, c, ldc, alpha, beta);
            else if (kind == KernelKind.Avx2) Kernel6x16(kc, pa, pb, c, ldc, alpha, beta);
            else KernelPortable(kc, pa, pb, c, ldc, alpha, beta);
        }

        // ------------------------------------------------------------------------------------
        //  Packing. A block -> MR-row panels laid out [p][r]; B block -> NR-column panels [p][j].
        //  Both zero padded to the full register tile. The layout the operand already has picks
        //  the loop: a contiguous run along the packed dimension is a straight vector copy, a
        //  run along K (row-major A, or B given as its transpose) goes through 8x8 transposes.
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackA(int kc, int mc, float* a, long ars, long acs, float* dst, int MR)
        {
            for (int i = 0; i < mc; i += MR)
            {
                int mr = Math.Min(MR, mc - i);
                float* src = a + i * ars;
                if (acs == 1 && mr == MR && (MR & 7) == 0 && Avx.IsSupported)
                {
                    PackTransposed(kc, src, ars, MR, dst, MR);
                }
                else if (acs == 1 && mr == MR && (MR == 4 || MR == 6) && Sse.IsSupported)
                {
                    PackRows4or6(kc, src, ars, MR, dst);
                }
                else if (ars == 1)
                {
                    for (int p = 0; p < kc; p++)
                    {
                        float* s = src + p * acs;
                        float* d = dst + p * MR;
                        int r = 0;
                        for (; r < mr; r++) d[r] = s[r];
                        for (; r < MR; r++) d[r] = 0f;
                    }
                }
                else
                {
                    for (int r = 0; r < mr; r++)
                    {
                        float* s = src + r * ars;
                        float* d = dst + r;
                        if (acs == 1)
                        {
                            for (int p = 0; p < kc; p++) d[p * MR] = s[p];
                        }
                        else
                        {
                            for (int p = 0; p < kc; p++) d[p * MR] = s[p * acs];
                        }
                    }
                    for (int r = mr; r < MR; r++)
                    {
                        float* d = dst + r;
                        for (int p = 0; p < kc; p++) d[p * MR] = 0f;
                    }
                }
                dst += (long)MR * kc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackB(int kc, int nc, float* b, long brs, long bcs, float* dst, int NR)
        {
            for (int j = 0; j < nc; j += NR)
            {
                int nr = Math.Min(NR, nc - j);
                float* src = b + j * bcs;
                if (bcs == 1)
                {
                    if (nr == NR)
                    {
                        for (int p = 0; p < kc; p++)
                            CopyPanelRow(src + p * brs, dst + p * NR, NR);
                    }
                    else
                    {
                        for (int p = 0; p < kc; p++)
                        {
                            float* s = src + p * brs;
                            float* d = dst + p * NR;
                            int q = 0;
                            for (; q < nr; q++) d[q] = s[q];
                            for (; q < NR; q++) d[q] = 0f;
                        }
                    }
                }
                else if (brs == 1 && nr == NR && (NR & 7) == 0 && Avx.IsSupported)
                {
                    PackTransposed(kc, src, bcs, NR, dst, NR);
                }
                else
                {
                    for (int q = 0; q < nr; q++)
                    {
                        float* s = src + q * bcs;
                        float* d = dst + q;
                        if (brs == 1)
                        {
                            for (int p = 0; p < kc; p++) d[p * NR] = s[p];
                        }
                        else
                        {
                            for (int p = 0; p < kc; p++) d[p * NR] = s[p * brs];
                        }
                    }
                    for (int q = nr; q < NR; q++)
                    {
                        float* d = dst + q;
                        for (int p = 0; p < kc; p++) d[p * NR] = 0f;
                    }
                }
                dst += (long)NR * kc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyPanelRow(float* src, float* dst, int count)
        {
            int q = 0;
            if (Vector512.IsHardwareAccelerated)
            {
                for (; q + 16 <= count; q += 16)
                    Vector512.Store(Vector512.Load(src + q), dst + q);
            }
            if (Vector256.IsHardwareAccelerated)
            {
                for (; q + 8 <= count; q += 8)
                    Vector256.Store(Vector256.Load(src + q), dst + q);
            }
            if (Vector128.IsHardwareAccelerated)
            {
                for (; q + 4 <= count; q += 4)
                    Vector128.Store(Vector128.Load(src + q), dst + q);
            }
            for (; q < count; q++)
                dst[q] = src[q];
        }

        /// <summary>
        /// <paramref name="lines"/> source lines (line q at src + q*lineStride, contiguous along p)
        /// -> dst[p*width + q]: the panel layout for a row-major A block or a transposed B block.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackTransposed(int kc, float* src, long lineStride, int lines, float* dst, int width)
        {
            for (int q0 = 0; q0 < lines; q0 += 8)
            {
                float* s0 = src + q0 * lineStride;
                float* s1 = s0 + lineStride;
                float* s2 = s1 + lineStride;
                float* s3 = s2 + lineStride;
                float* s4 = s3 + lineStride;
                float* s5 = s4 + lineStride;
                float* s6 = s5 + lineStride;
                float* s7 = s6 + lineStride;
                float* d = dst + q0;
                int p = 0;
                for (; p + 8 <= kc; p += 8)
                {
                    Transpose8x8(s0 + p, s1 + p, s2 + p, s3 + p, s4 + p, s5 + p, s6 + p, s7 + p, d + p * width, width);
                }
                for (; p < kc; p++)
                {
                    float* dp = d + p * width;
                    dp[0] = s0[p]; dp[1] = s1[p]; dp[2] = s2[p]; dp[3] = s3[p];
                    dp[4] = s4[p]; dp[5] = s5[p]; dp[6] = s6[p]; dp[7] = s7[p];
                }
            }
        }

        /// <summary>
        /// Row-major A panel of 4 or 6 rows (the portable and AVX2 6x16 tiles) -> dst[p*MR + r]
        /// with SSE 4x4 transposes (+ a 2-row interleave for rows 4-5). The scalar strided loop
        /// this replaces cost about as much as the arithmetic of an N = 64 product.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void PackRows4or6(int kc, float* src, long lineStride, int MR, float* dst)
        {
            float* s0 = src;
            float* s1 = s0 + lineStride;
            float* s2 = s1 + lineStride;
            float* s3 = s2 + lineStride;
            float* s4 = MR == 6 ? s3 + lineStride : s0;
            float* s5 = MR == 6 ? s4 + lineStride : s0;
            int p = 0;
            for (; p + 4 <= kc; p += 4)
            {
                Vector128<float> r0 = Sse.LoadVector128(s0 + p), r1 = Sse.LoadVector128(s1 + p);
                Vector128<float> r2 = Sse.LoadVector128(s2 + p), r3 = Sse.LoadVector128(s3 + p);
                Vector128<float> t0 = Sse.UnpackLow(r0, r1), t1 = Sse.UnpackHigh(r0, r1);
                Vector128<float> t2 = Sse.UnpackLow(r2, r3), t3 = Sse.UnpackHigh(r2, r3);
                float* d = dst + p * MR;
                Sse.Store(d, Sse.MoveLowToHigh(t0, t2));             // rows 0-3 at p
                Sse.Store(d + MR, Sse.MoveHighToLow(t2, t0));        // p + 1
                Sse.Store(d + 2 * MR, Sse.MoveLowToHigh(t1, t3));    // p + 2
                Sse.Store(d + 3 * MR, Sse.MoveHighToLow(t3, t1));    // p + 3
                if (MR == 6)
                {
                    Vector128<float> r4 = Sse.LoadVector128(s4 + p), r5 = Sse.LoadVector128(s5 + p);
                    Vector128<float> lo = Sse.UnpackLow(r4, r5), hi = Sse.UnpackHigh(r4, r5);
                    Sse.StoreLow(d + 4, lo);                         // rows 4-5 at p
                    Sse.StoreHigh(d + MR + 4, lo);
                    Sse.StoreLow(d + 2 * MR + 4, hi);
                    Sse.StoreHigh(d + 3 * MR + 4, hi);
                }
            }
            for (; p < kc; p++)
            {
                float* d = dst + p * MR;
                d[0] = s0[p]; d[1] = s1[p]; d[2] = s2[p]; d[3] = s3[p];
                if (MR == 6)
                {
                    d[4] = s4[p];
                    d[5] = s5[p];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Transpose8x8(float* s0, float* s1, float* s2, float* s3,
            float* s4, float* s5, float* s6, float* s7, float* d, long dstStride)
        {
            Vector256<float> r0 = Avx.LoadVector256(s0), r1 = Avx.LoadVector256(s1);
            Vector256<float> r2 = Avx.LoadVector256(s2), r3 = Avx.LoadVector256(s3);
            Vector256<float> r4 = Avx.LoadVector256(s4), r5 = Avx.LoadVector256(s5);
            Vector256<float> r6 = Avx.LoadVector256(s6), r7 = Avx.LoadVector256(s7);

            Vector256<float> t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
            Vector256<float> t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
            Vector256<float> t4 = Avx.UnpackLow(r4, r5), t5 = Avx.UnpackHigh(r4, r5);
            Vector256<float> t6 = Avx.UnpackLow(r6, r7), t7 = Avx.UnpackHigh(r6, r7);

            Vector256<float> u0 = Avx.Shuffle(t0, t2, 0x44), u1 = Avx.Shuffle(t0, t2, 0xEE);
            Vector256<float> u2 = Avx.Shuffle(t1, t3, 0x44), u3 = Avx.Shuffle(t1, t3, 0xEE);
            Vector256<float> u4 = Avx.Shuffle(t4, t6, 0x44), u5 = Avx.Shuffle(t4, t6, 0xEE);
            Vector256<float> u6 = Avx.Shuffle(t5, t7, 0x44), u7 = Avx.Shuffle(t5, t7, 0xEE);

            Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
            Avx.Store(d + dstStride, Avx.Permute2x128(u1, u5, 0x20));
            Avx.Store(d + 2 * dstStride, Avx.Permute2x128(u2, u6, 0x20));
            Avx.Store(d + 3 * dstStride, Avx.Permute2x128(u3, u7, 0x20));
            Avx.Store(d + 4 * dstStride, Avx.Permute2x128(u0, u4, 0x31));
            Avx.Store(d + 5 * dstStride, Avx.Permute2x128(u1, u5, 0x31));
            Avx.Store(d + 6 * dstStride, Avx.Permute2x128(u2, u6, 0x31));
            Avx.Store(d + 7 * dstStride, Avx.Permute2x128(u3, u7, 0x31));
        }

        // ------------------------------------------------------------------------------------
        //  Microkernels: C[MR x NR] (op)= alpha * sum_p Apanel[p][0..MR) (x) Bpanel[p][0..NR).
        //  beta == 0 stores without reading C, beta == 1 accumulates (later K blocks).
        // ------------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel8x32(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector512<float> c00 = Vector512<float>.Zero, c01 = Vector512<float>.Zero;
            Vector512<float> c10 = Vector512<float>.Zero, c11 = Vector512<float>.Zero;
            Vector512<float> c20 = Vector512<float>.Zero, c21 = Vector512<float>.Zero;
            Vector512<float> c30 = Vector512<float>.Zero, c31 = Vector512<float>.Zero;
            Vector512<float> c40 = Vector512<float>.Zero, c41 = Vector512<float>.Zero;
            Vector512<float> c50 = Vector512<float>.Zero, c51 = Vector512<float>.Zero;
            Vector512<float> c60 = Vector512<float>.Zero, c61 = Vector512<float>.Zero;
            Vector512<float> c70 = Vector512<float>.Zero, c71 = Vector512<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector512<float> b0 = Avx512F.LoadVector512(pb);
                Vector512<float> b1 = Avx512F.LoadVector512(pb + 16);
                Vector512<float> av = Vector512.Create(pa[0]);
                c00 = Avx512F.FusedMultiplyAdd(av, b0, c00);
                c01 = Avx512F.FusedMultiplyAdd(av, b1, c01);
                av = Vector512.Create(pa[1]);
                c10 = Avx512F.FusedMultiplyAdd(av, b0, c10);
                c11 = Avx512F.FusedMultiplyAdd(av, b1, c11);
                av = Vector512.Create(pa[2]);
                c20 = Avx512F.FusedMultiplyAdd(av, b0, c20);
                c21 = Avx512F.FusedMultiplyAdd(av, b1, c21);
                av = Vector512.Create(pa[3]);
                c30 = Avx512F.FusedMultiplyAdd(av, b0, c30);
                c31 = Avx512F.FusedMultiplyAdd(av, b1, c31);
                av = Vector512.Create(pa[4]);
                c40 = Avx512F.FusedMultiplyAdd(av, b0, c40);
                c41 = Avx512F.FusedMultiplyAdd(av, b1, c41);
                av = Vector512.Create(pa[5]);
                c50 = Avx512F.FusedMultiplyAdd(av, b0, c50);
                c51 = Avx512F.FusedMultiplyAdd(av, b1, c51);
                av = Vector512.Create(pa[6]);
                c60 = Avx512F.FusedMultiplyAdd(av, b0, c60);
                c61 = Avx512F.FusedMultiplyAdd(av, b1, c61);
                av = Vector512.Create(pa[7]);
                c70 = Avx512F.FusedMultiplyAdd(av, b0, c70);
                c71 = Avx512F.FusedMultiplyAdd(av, b1, c71);
                pa += 8;
                pb += 32;
            }

            Vector512<float> va = Vector512.Create(alpha);
            if (beta == 0f)
            {
                Store512x2(c, c00, c01, va); Store512x2(c + ldc, c10, c11, va);
                Store512x2(c + 2 * ldc, c20, c21, va); Store512x2(c + 3 * ldc, c30, c31, va);
                Store512x2(c + 4 * ldc, c40, c41, va); Store512x2(c + 5 * ldc, c50, c51, va);
                Store512x2(c + 6 * ldc, c60, c61, va); Store512x2(c + 7 * ldc, c70, c71, va);
            }
            else
            {
                Vector512<float> vb = Vector512.Create(beta);
                bool one = beta == 1f;
                Update512x2(c, c00, c01, va, vb, one); Update512x2(c + ldc, c10, c11, va, vb, one);
                Update512x2(c + 2 * ldc, c20, c21, va, vb, one); Update512x2(c + 3 * ldc, c30, c31, va, vb, one);
                Update512x2(c + 4 * ldc, c40, c41, va, vb, one); Update512x2(c + 5 * ldc, c50, c51, va, vb, one);
                Update512x2(c + 6 * ldc, c60, c61, va, vb, one); Update512x2(c + 7 * ldc, c70, c71, va, vb, one);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store512x2(float* c, Vector512<float> v0, Vector512<float> v1, Vector512<float> va)
        {
            Avx512F.Store(c, Avx512F.Multiply(v0, va));
            Avx512F.Store(c + 16, Avx512F.Multiply(v1, va));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Update512x2(float* c, Vector512<float> v0, Vector512<float> v1,
            Vector512<float> va, Vector512<float> vb, bool betaIsOne)
        {
            Vector512<float> o0 = Avx512F.LoadVector512(c);
            Vector512<float> o1 = Avx512F.LoadVector512(c + 16);
            if (!betaIsOne)
            {
                o0 = Avx512F.Multiply(o0, vb);
                o1 = Avx512F.Multiply(o1, vb);
            }
            Avx512F.Store(c, Avx512F.FusedMultiplyAdd(v0, va, o0));
            Avx512F.Store(c + 16, Avx512F.FusedMultiplyAdd(v1, va, o1));
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel6x16(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector256<float> c00 = Vector256<float>.Zero, c01 = Vector256<float>.Zero;
            Vector256<float> c10 = Vector256<float>.Zero, c11 = Vector256<float>.Zero;
            Vector256<float> c20 = Vector256<float>.Zero, c21 = Vector256<float>.Zero;
            Vector256<float> c30 = Vector256<float>.Zero, c31 = Vector256<float>.Zero;
            Vector256<float> c40 = Vector256<float>.Zero, c41 = Vector256<float>.Zero;
            Vector256<float> c50 = Vector256<float>.Zero, c51 = Vector256<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector256<float> b0 = Avx.LoadVector256(pb);
                Vector256<float> b1 = Avx.LoadVector256(pb + 8);
                Vector256<float> av = Vector256.Create(pa[0]);
                c00 = Fma.MultiplyAdd(av, b0, c00);
                c01 = Fma.MultiplyAdd(av, b1, c01);
                av = Vector256.Create(pa[1]);
                c10 = Fma.MultiplyAdd(av, b0, c10);
                c11 = Fma.MultiplyAdd(av, b1, c11);
                av = Vector256.Create(pa[2]);
                c20 = Fma.MultiplyAdd(av, b0, c20);
                c21 = Fma.MultiplyAdd(av, b1, c21);
                av = Vector256.Create(pa[3]);
                c30 = Fma.MultiplyAdd(av, b0, c30);
                c31 = Fma.MultiplyAdd(av, b1, c31);
                av = Vector256.Create(pa[4]);
                c40 = Fma.MultiplyAdd(av, b0, c40);
                c41 = Fma.MultiplyAdd(av, b1, c41);
                av = Vector256.Create(pa[5]);
                c50 = Fma.MultiplyAdd(av, b0, c50);
                c51 = Fma.MultiplyAdd(av, b1, c51);
                pa += 6;
                pb += 16;
            }

            Vector256<float> va = Vector256.Create(alpha);
            if (beta == 0f)
            {
                Store256x2(c, c00, c01, va); Store256x2(c + ldc, c10, c11, va);
                Store256x2(c + 2 * ldc, c20, c21, va); Store256x2(c + 3 * ldc, c30, c31, va);
                Store256x2(c + 4 * ldc, c40, c41, va); Store256x2(c + 5 * ldc, c50, c51, va);
            }
            else
            {
                Vector256<float> vb = Vector256.Create(beta);
                bool one = beta == 1f;
                Update256x2(c, c00, c01, va, vb, one); Update256x2(c + ldc, c10, c11, va, vb, one);
                Update256x2(c + 2 * ldc, c20, c21, va, vb, one); Update256x2(c + 3 * ldc, c30, c31, va, vb, one);
                Update256x2(c + 4 * ldc, c40, c41, va, vb, one); Update256x2(c + 5 * ldc, c50, c51, va, vb, one);
            }
        }

        /// <summary>
        /// 8x24 tile on 256-bit vectors: 24 ymm accumulators, which needs the 32-register EVEX
        /// file (AVX-512VL hardware). Same register-tile volume as the zmm kernel, but 256-bit
        /// FMAs keep client cores in their AVX2 turbo licence.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static void Kernel8x24(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            Vector256<float> c00 = Vector256<float>.Zero, c01 = Vector256<float>.Zero, c02 = Vector256<float>.Zero;
            Vector256<float> c10 = Vector256<float>.Zero, c11 = Vector256<float>.Zero, c12 = Vector256<float>.Zero;
            Vector256<float> c20 = Vector256<float>.Zero, c21 = Vector256<float>.Zero, c22 = Vector256<float>.Zero;
            Vector256<float> c30 = Vector256<float>.Zero, c31 = Vector256<float>.Zero, c32 = Vector256<float>.Zero;
            Vector256<float> c40 = Vector256<float>.Zero, c41 = Vector256<float>.Zero, c42 = Vector256<float>.Zero;
            Vector256<float> c50 = Vector256<float>.Zero, c51 = Vector256<float>.Zero, c52 = Vector256<float>.Zero;
            Vector256<float> c60 = Vector256<float>.Zero, c61 = Vector256<float>.Zero, c62 = Vector256<float>.Zero;
            Vector256<float> c70 = Vector256<float>.Zero, c71 = Vector256<float>.Zero, c72 = Vector256<float>.Zero;

            for (int p = 0; p < kc; p++)
            {
                Vector256<float> b0 = Avx.LoadVector256(pb);
                Vector256<float> b1 = Avx.LoadVector256(pb + 8);
                Vector256<float> b2 = Avx.LoadVector256(pb + 16);
                Vector256<float> av = Vector256.Create(pa[0]);
                c00 = Fma.MultiplyAdd(av, b0, c00); c01 = Fma.MultiplyAdd(av, b1, c01); c02 = Fma.MultiplyAdd(av, b2, c02);
                av = Vector256.Create(pa[1]);
                c10 = Fma.MultiplyAdd(av, b0, c10); c11 = Fma.MultiplyAdd(av, b1, c11); c12 = Fma.MultiplyAdd(av, b2, c12);
                av = Vector256.Create(pa[2]);
                c20 = Fma.MultiplyAdd(av, b0, c20); c21 = Fma.MultiplyAdd(av, b1, c21); c22 = Fma.MultiplyAdd(av, b2, c22);
                av = Vector256.Create(pa[3]);
                c30 = Fma.MultiplyAdd(av, b0, c30); c31 = Fma.MultiplyAdd(av, b1, c31); c32 = Fma.MultiplyAdd(av, b2, c32);
                av = Vector256.Create(pa[4]);
                c40 = Fma.MultiplyAdd(av, b0, c40); c41 = Fma.MultiplyAdd(av, b1, c41); c42 = Fma.MultiplyAdd(av, b2, c42);
                av = Vector256.Create(pa[5]);
                c50 = Fma.MultiplyAdd(av, b0, c50); c51 = Fma.MultiplyAdd(av, b1, c51); c52 = Fma.MultiplyAdd(av, b2, c52);
                av = Vector256.Create(pa[6]);
                c60 = Fma.MultiplyAdd(av, b0, c60); c61 = Fma.MultiplyAdd(av, b1, c61); c62 = Fma.MultiplyAdd(av, b2, c62);
                av = Vector256.Create(pa[7]);
                c70 = Fma.MultiplyAdd(av, b0, c70); c71 = Fma.MultiplyAdd(av, b1, c71); c72 = Fma.MultiplyAdd(av, b2, c72);
                pa += 8;
                pb += 24;
            }

            Vector256<float> va = Vector256.Create(alpha);
            Vector256<float> vb = Vector256.Create(beta);
            Row256x3(c, c00, c01, c02, va, vb, beta);
            Row256x3(c + ldc, c10, c11, c12, va, vb, beta);
            Row256x3(c + 2 * ldc, c20, c21, c22, va, vb, beta);
            Row256x3(c + 3 * ldc, c30, c31, c32, va, vb, beta);
            Row256x3(c + 4 * ldc, c40, c41, c42, va, vb, beta);
            Row256x3(c + 5 * ldc, c50, c51, c52, va, vb, beta);
            Row256x3(c + 6 * ldc, c60, c61, c62, va, vb, beta);
            Row256x3(c + 7 * ldc, c70, c71, c72, va, vb, beta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Row256x3(float* c, Vector256<float> v0, Vector256<float> v1, Vector256<float> v2,
            Vector256<float> va, Vector256<float> vb, float beta)
        {
            if (beta == 0f)
            {
                Avx.Store(c, Avx.Multiply(v0, va));
                Avx.Store(c + 8, Avx.Multiply(v1, va));
                Avx.Store(c + 16, Avx.Multiply(v2, va));
                return;
            }

            Vector256<float> o0 = Avx.LoadVector256(c);
            Vector256<float> o1 = Avx.LoadVector256(c + 8);
            Vector256<float> o2 = Avx.LoadVector256(c + 16);
            if (beta != 1f)
            {
                o0 = Avx.Multiply(o0, vb);
                o1 = Avx.Multiply(o1, vb);
                o2 = Avx.Multiply(o2, vb);
            }
            Avx.Store(c, Fma.MultiplyAdd(v0, va, o0));
            Avx.Store(c + 8, Fma.MultiplyAdd(v1, va, o1));
            Avx.Store(c + 16, Fma.MultiplyAdd(v2, va, o2));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Store256x2(float* c, Vector256<float> v0, Vector256<float> v1, Vector256<float> va)
        {
            Avx.Store(c, Avx.Multiply(v0, va));
            Avx.Store(c + 8, Avx.Multiply(v1, va));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Update256x2(float* c, Vector256<float> v0, Vector256<float> v1,
            Vector256<float> va, Vector256<float> vb, bool betaIsOne)
        {
            Vector256<float> o0 = Avx.LoadVector256(c);
            Vector256<float> o1 = Avx.LoadVector256(c + 8);
            if (!betaIsOne)
            {
                o0 = Avx.Multiply(o0, vb);
                o1 = Avx.Multiply(o1, vb);
            }
            Avx.Store(c, Fma.MultiplyAdd(v0, va, o0));
            Avx.Store(c + 8, Fma.MultiplyAdd(v1, va, o1));
        }

        /// <summary>Portable 4 x (2 * Vector&lt;float&gt;.Count) tile (ARM64 NEON, pre-AVX2 x64).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void KernelPortable(int kc, float* pa, float* pb, float* c, long ldc, float alpha, float beta)
        {
            int w = Vector<float>.Count;
            Vector<float> c00 = Vector<float>.Zero, c01 = Vector<float>.Zero;
            Vector<float> c10 = Vector<float>.Zero, c11 = Vector<float>.Zero;
            Vector<float> c20 = Vector<float>.Zero, c21 = Vector<float>.Zero;
            Vector<float> c30 = Vector<float>.Zero, c31 = Vector<float>.Zero;
            for (int p = 0; p < kc; p++)
            {
                Vector<float> b0 = Unsafe.ReadUnaligned<Vector<float>>(pb);
                Vector<float> b1 = Unsafe.ReadUnaligned<Vector<float>>(pb + w);
                Vector<float> av = new Vector<float>(pa[0]);
                c00 += av * b0; c01 += av * b1;
                av = new Vector<float>(pa[1]);
                c10 += av * b0; c11 += av * b1;
                av = new Vector<float>(pa[2]);
                c20 += av * b0; c21 += av * b1;
                av = new Vector<float>(pa[3]);
                c30 += av * b0; c31 += av * b1;
                pa += 4;
                pb += 2 * w;
            }

            StorePortable(c, c00, c01, alpha, beta, w);
            StorePortable(c + ldc, c10, c11, alpha, beta, w);
            StorePortable(c + 2 * ldc, c20, c21, alpha, beta, w);
            StorePortable(c + 3 * ldc, c30, c31, alpha, beta, w);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StorePortable(float* c, Vector<float> v0, Vector<float> v1, float alpha, float beta, int w)
        {
            Vector<float> va = new Vector<float>(alpha);
            v0 *= va;
            v1 *= va;
            if (beta != 0f)
            {
                Vector<float> o0 = Unsafe.ReadUnaligned<Vector<float>>(c);
                Vector<float> o1 = Unsafe.ReadUnaligned<Vector<float>>(c + w);
                if (beta == 1f)
                {
                    v0 += o0;
                    v1 += o1;
                }
                else
                {
                    Vector<float> vb = new Vector<float>(beta);
                    v0 += o0 * vb;
                    v1 += o1 * vb;
                }
            }
            Unsafe.WriteUnaligned(c, v0);
            Unsafe.WriteUnaligned(c + w, v1);
        }

        // ------------------------------------------------------------------------------------
        //  Narrow dot products: N small, A rows and B columns both contiguous along K (A times a
        //  transposed B: DirectOps.CpuGemmABt, Addmm against W^T, attention over few keys). The
        //  packed path pads N up to the register tile (a 3-channel conv_out wastes 29/32 of an
        //  8x32 tile's FMAs) and packs every A element for only N uses. Here each 4-row block
        //  runs K-long dot products against 4 columns at a time (2 with 256-bit and portable
        //  vectors, to fit 16 registers) straight from the operands: A streams from memory once, its 4
        //  rows stay in L1 across the column groups and B^T (N x K) stays in L2.
        // ------------------------------------------------------------------------------------

        // B^T (N x K) is re-read from cache by every 4-row block, so K is cut into blocks when it
        // would outgrow this share of L2. Not otherwise: a K split makes each A row two passes
        // apart, and that costs DRAM bandwidth (N = 16, K = 2592 split in two ran 17% slower).
        private const long DotBBudgetBytes = 512 * 1024;

        private static int DotLanes(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => 16,
            KernelKind.Avx2 or KernelKind.Avx2Wide => 8,
            _ => Vector<float>.Count,
        };

        // Crossover with the packed tile, from CpuFloatBench 'narrow' (i7-11800H; M = 300..65536,
        // K = 288..4096): the zmm dot tile beats the 8x32 tile through N = 64 (mixed at 96), the
        // 4x2 ymm tile beats 6x16 through N = 64 and the EVEX 8x24 through N = 40, the portable
        // one on 256-bit vectors through N = 40. With 128-bit vectors (ARM64) the portable
        // packed tile is only 8 columns wide, so little is padded beyond N = 8; unmeasured there,
        // the dot path stays below that.
        private static int DefaultNarrowDotMaxN(KernelKind kind) => kind switch
        {
            KernelKind.Avx512 => 64,
            KernelKind.Avx2Wide => 40,
            KernelKind.Avx2 => 64,
            _ => Vector<float>.Count >= 8 ? 40 : 7,
        };

        private static void NarrowDot(int batch, int m, int n, int k, float alpha,
            float* a, long aBatchStride, long ars, float* b, long bBatchStride, long bcs,
            float beta, float* c, long cBatchStride, long ldc, int threads, double totalFlops)
        {
            KernelKind kind = _kernel;
            if (threads <= 1 || totalFlops < ParallelFlopThreshold)
            {
                for (int bi = 0; bi < batch; bi++)
                {
                    DotRows(kind, m, n, k, alpha, a + bi * aBatchStride, ars, b + bi * bBatchStride, bcs,
                        beta, c + bi * cBatchStride, ldc);
                }
                return;
            }

            // A few row chunks per thread (the pool hands them out dynamically), each worth at
            // least MinTileFlops; chunk edges on the 4-row blocks.
            int rowBlocks = CeilDiv(m, 4);
            int maxChunks = (int)Math.Max(1, Math.Min(4.0 * threads, totalFlops / MinTileFlops));
            int chunksPerItem = Math.Clamp(CeilDiv(maxChunks, batch), 1, rowBlocks);
            int rowsPerChunk = CeilDiv(rowBlocks, chunksPerItem) * 4;
            chunksPerItem = CeilDiv(m, rowsPerChunk);
            CpuParallel.For(batch * chunksPerItem, t =>
            {
                int bi = t / chunksPerItem;
                int r0 = (t - bi * chunksPerItem) * rowsPerChunk;
                DotRows(kind, Math.Min(rowsPerChunk, m - r0), n, k, alpha,
                    a + bi * aBatchStride + r0 * ars, ars, b + bi * bBatchStride, bcs,
                    beta, c + bi * cBatchStride + r0 * ldc, ldc);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DotRows(KernelKind kind, int m, int n, int k, float alpha,
            float* a, long ars, float* b, long bcs, float beta, float* c, long ldc)
        {
            int cols = kind == KernelKind.Avx512 ? 4 : 2;
            // Even K blocks on a 16-element grid, so only the last block has a vector tail. N
            // within one column group keeps B^T to a few rows, so that case never splits.
            long bBytes = (long)n * k * sizeof(float);
            int kBlocks = n > cols && bBytes > DotBBudgetBytes ? (int)((bBytes + DotBBudgetBytes - 1) / DotBBudgetBytes) : 1;
            int kd = kBlocks == 1 ? k : RoundUp(CeilDiv(k, kBlocks), 16);
            for (int p0 = 0; p0 < k; p0 += kd)
            {
                int len = Math.Min(kd, k - p0);
                float betaEff = p0 == 0 ? beta : 1f;
                for (int i = 0; i < m; i += 4)
                {
                    int rows = Math.Min(4, m - i);
                    float* ai = a + i * ars + p0;
                    float* ci = c + i * ldc;
                    for (int j = 0; j < n; j += cols)
                    {
                        int nc = Math.Min(cols, n - j);
                        float* bj = b + j * bcs + p0;
                        if (kind == KernelKind.Avx512) DotTile512(len, ai, ars, rows, bj, bcs, nc, alpha, betaEff, ci + j, ldc);
                        else if (kind == KernelKind.Portable) DotTilePortable(len, ai, ars, rows, bj, bcs, nc, alpha, betaEff, ci + j, ldc);
                        else DotTile256(len, ai, ars, rows, bj, bcs, nc, alpha, betaEff, ci + j, ldc);
                    }
                }
            }
        }

        // The dot tiles below share two tricks. Rows/columns missing from an edge tile alias the
        // first one and their sums are dropped. A K tail is one more step over the LAST full
        // vector of the row (it may start in an earlier K block: DotRows only runs with
        // K >= lanes) with the lanes already summed zeroed in both operands, so an Inf/NaN there
        // cannot re-enter as 0 * Inf.

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DotTile512(int k, float* a, long ars, int rows, float* b, long bcs, int cols,
            float alpha, float beta, float* c, long ldc)
        {
            float* a0 = a, a1 = rows > 1 ? a + ars : a, a2 = rows > 2 ? a + 2 * ars : a, a3 = rows > 3 ? a + 3 * ars : a;
            float* b0 = b, b1 = cols > 1 ? b + bcs : b, b2 = cols > 2 ? b + 2 * bcs : b, b3 = cols > 3 ? b + 3 * bcs : b;
            Vector512<float> s00 = default, s01 = default, s02 = default, s03 = default;
            Vector512<float> s10 = default, s11 = default, s12 = default, s13 = default;
            Vector512<float> s20 = default, s21 = default, s22 = default, s23 = default;
            Vector512<float> s30 = default, s31 = default, s32 = default, s33 = default;
            Vector512<float> x0, x1, x2, x3, y;

            int p = 0;
            for (; p + 16 <= k; p += 16)
            {
                x0 = Vector512.Load(a0 + p); x1 = Vector512.Load(a1 + p);
                x2 = Vector512.Load(a2 + p); x3 = Vector512.Load(a3 + p);
                y = Vector512.Load(b0 + p);
                s00 = Avx512F.FusedMultiplyAdd(x0, y, s00); s10 = Avx512F.FusedMultiplyAdd(x1, y, s10);
                s20 = Avx512F.FusedMultiplyAdd(x2, y, s20); s30 = Avx512F.FusedMultiplyAdd(x3, y, s30);
                y = Vector512.Load(b1 + p);
                s01 = Avx512F.FusedMultiplyAdd(x0, y, s01); s11 = Avx512F.FusedMultiplyAdd(x1, y, s11);
                s21 = Avx512F.FusedMultiplyAdd(x2, y, s21); s31 = Avx512F.FusedMultiplyAdd(x3, y, s31);
                y = Vector512.Load(b2 + p);
                s02 = Avx512F.FusedMultiplyAdd(x0, y, s02); s12 = Avx512F.FusedMultiplyAdd(x1, y, s12);
                s22 = Avx512F.FusedMultiplyAdd(x2, y, s22); s32 = Avx512F.FusedMultiplyAdd(x3, y, s32);
                y = Vector512.Load(b3 + p);
                s03 = Avx512F.FusedMultiplyAdd(x0, y, s03); s13 = Avx512F.FusedMultiplyAdd(x1, y, s13);
                s23 = Avx512F.FusedMultiplyAdd(x2, y, s23); s33 = Avx512F.FusedMultiplyAdd(x3, y, s33);
            }
            if (p < k)
            {
                Vector512<float> mask = Vector512.GreaterThanOrEqual(Vector512<int>.Indices, Vector512.Create(16 - (k - p))).AsSingle();
                p = k - 16;
                x0 = Vector512.Load(a0 + p) & mask; x1 = Vector512.Load(a1 + p) & mask;
                x2 = Vector512.Load(a2 + p) & mask; x3 = Vector512.Load(a3 + p) & mask;
                y = Vector512.Load(b0 + p) & mask;
                s00 = Avx512F.FusedMultiplyAdd(x0, y, s00); s10 = Avx512F.FusedMultiplyAdd(x1, y, s10);
                s20 = Avx512F.FusedMultiplyAdd(x2, y, s20); s30 = Avx512F.FusedMultiplyAdd(x3, y, s30);
                y = Vector512.Load(b1 + p) & mask;
                s01 = Avx512F.FusedMultiplyAdd(x0, y, s01); s11 = Avx512F.FusedMultiplyAdd(x1, y, s11);
                s21 = Avx512F.FusedMultiplyAdd(x2, y, s21); s31 = Avx512F.FusedMultiplyAdd(x3, y, s31);
                y = Vector512.Load(b2 + p) & mask;
                s02 = Avx512F.FusedMultiplyAdd(x0, y, s02); s12 = Avx512F.FusedMultiplyAdd(x1, y, s12);
                s22 = Avx512F.FusedMultiplyAdd(x2, y, s22); s32 = Avx512F.FusedMultiplyAdd(x3, y, s32);
                y = Vector512.Load(b3 + p) & mask;
                s03 = Avx512F.FusedMultiplyAdd(x0, y, s03); s13 = Avx512F.FusedMultiplyAdd(x1, y, s13);
                s23 = Avx512F.FusedMultiplyAdd(x2, y, s23); s33 = Avx512F.FusedMultiplyAdd(x3, y, s33);
            }

            StoreDotRow(c, Reduce4(Fold(s00), Fold(s01), Fold(s02), Fold(s03)), cols, alpha, beta);
            if (rows > 1) StoreDotRow(c + ldc, Reduce4(Fold(s10), Fold(s11), Fold(s12), Fold(s13)), cols, alpha, beta);
            if (rows > 2) StoreDotRow(c + 2 * ldc, Reduce4(Fold(s20), Fold(s21), Fold(s22), Fold(s23)), cols, alpha, beta);
            if (rows > 3) StoreDotRow(c + 3 * ldc, Reduce4(Fold(s30), Fold(s31), Fold(s32), Fold(s33)), cols, alpha, beta);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DotTile256(int k, float* a, long ars, int rows, float* b, long bcs, int cols,
            float alpha, float beta, float* c, long ldc)
        {
            float* a0 = a, a1 = rows > 1 ? a + ars : a, a2 = rows > 2 ? a + 2 * ars : a, a3 = rows > 3 ? a + 3 * ars : a;
            float* b0 = b, b1 = cols > 1 ? b + bcs : b;
            Vector256<float> s00 = default, s01 = default, s10 = default, s11 = default;
            Vector256<float> s20 = default, s21 = default, s30 = default, s31 = default;
            Vector256<float> x0, x1, x2, x3, y;

            int p = 0;
            for (; p + 8 <= k; p += 8)
            {
                x0 = Vector256.Load(a0 + p); x1 = Vector256.Load(a1 + p);
                x2 = Vector256.Load(a2 + p); x3 = Vector256.Load(a3 + p);
                y = Vector256.Load(b0 + p);
                s00 = Fma.MultiplyAdd(x0, y, s00); s10 = Fma.MultiplyAdd(x1, y, s10);
                s20 = Fma.MultiplyAdd(x2, y, s20); s30 = Fma.MultiplyAdd(x3, y, s30);
                y = Vector256.Load(b1 + p);
                s01 = Fma.MultiplyAdd(x0, y, s01); s11 = Fma.MultiplyAdd(x1, y, s11);
                s21 = Fma.MultiplyAdd(x2, y, s21); s31 = Fma.MultiplyAdd(x3, y, s31);
            }
            if (p < k)
            {
                Vector256<float> mask = Vector256.GreaterThanOrEqual(Vector256<int>.Indices, Vector256.Create(8 - (k - p))).AsSingle();
                p = k - 8;
                x0 = Vector256.Load(a0 + p) & mask; x1 = Vector256.Load(a1 + p) & mask;
                x2 = Vector256.Load(a2 + p) & mask; x3 = Vector256.Load(a3 + p) & mask;
                y = Vector256.Load(b0 + p) & mask;
                s00 = Fma.MultiplyAdd(x0, y, s00); s10 = Fma.MultiplyAdd(x1, y, s10);
                s20 = Fma.MultiplyAdd(x2, y, s20); s30 = Fma.MultiplyAdd(x3, y, s30);
                y = Vector256.Load(b1 + p) & mask;
                s01 = Fma.MultiplyAdd(x0, y, s01); s11 = Fma.MultiplyAdd(x1, y, s11);
                s21 = Fma.MultiplyAdd(x2, y, s21); s31 = Fma.MultiplyAdd(x3, y, s31);
            }

            // Each reduction is [row r col 0, row r col 1, row r+1 col 0, row r+1 col 1].
            Vector128<float> r01 = Reduce4(s00, s01, s10, s11);
            Vector128<float> r23 = Reduce4(s20, s21, s30, s31);
            StoreDotPair(c, r01.GetElement(0), r01.GetElement(1), cols, alpha, beta);
            if (rows > 1) StoreDotPair(c + ldc, r01.GetElement(2), r01.GetElement(3), cols, alpha, beta);
            if (rows > 2) StoreDotPair(c + 2 * ldc, r23.GetElement(0), r23.GetElement(1), cols, alpha, beta);
            if (rows > 3) StoreDotPair(c + 3 * ldc, r23.GetElement(2), r23.GetElement(3), cols, alpha, beta);
        }

        /// <summary>Vector&lt;T&gt; form of the 4x2 dot tile (ARM64 NEON, pre-AVX2 x64).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void DotTilePortable(int k, float* a, long ars, int rows, float* b, long bcs, int cols,
            float alpha, float beta, float* c, long ldc)
        {
            int w = Vector<float>.Count;
            float* a0 = a, a1 = rows > 1 ? a + ars : a, a2 = rows > 2 ? a + 2 * ars : a, a3 = rows > 3 ? a + 3 * ars : a;
            float* b0 = b, b1 = cols > 1 ? b + bcs : b;
            Vector<float> s00 = default, s01 = default, s10 = default, s11 = default;
            Vector<float> s20 = default, s21 = default, s30 = default, s31 = default;
            Vector<float> x0, x1, x2, x3, y;

            int p = 0;
            for (; p + w <= k; p += w)
            {
                x0 = Unsafe.ReadUnaligned<Vector<float>>(a0 + p); x1 = Unsafe.ReadUnaligned<Vector<float>>(a1 + p);
                x2 = Unsafe.ReadUnaligned<Vector<float>>(a2 + p); x3 = Unsafe.ReadUnaligned<Vector<float>>(a3 + p);
                y = Unsafe.ReadUnaligned<Vector<float>>(b0 + p);
                s00 += x0 * y; s10 += x1 * y; s20 += x2 * y; s30 += x3 * y;
                y = Unsafe.ReadUnaligned<Vector<float>>(b1 + p);
                s01 += x0 * y; s11 += x1 * y; s21 += x2 * y; s31 += x3 * y;
            }
            if (p < k)
            {
                Vector<float> mask = Vector.AsVectorSingle(Vector.GreaterThanOrEqual(Vector<int>.Indices, new Vector<int>(w - (k - p))));
                p = k - w;
                x0 = Unsafe.ReadUnaligned<Vector<float>>(a0 + p) & mask; x1 = Unsafe.ReadUnaligned<Vector<float>>(a1 + p) & mask;
                x2 = Unsafe.ReadUnaligned<Vector<float>>(a2 + p) & mask; x3 = Unsafe.ReadUnaligned<Vector<float>>(a3 + p) & mask;
                y = Unsafe.ReadUnaligned<Vector<float>>(b0 + p) & mask;
                s00 += x0 * y; s10 += x1 * y; s20 += x2 * y; s30 += x3 * y;
                y = Unsafe.ReadUnaligned<Vector<float>>(b1 + p) & mask;
                s01 += x0 * y; s11 += x1 * y; s21 += x2 * y; s31 += x3 * y;
            }

            StoreDotPair(c, Vector.Sum(s00), Vector.Sum(s01), cols, alpha, beta);
            if (rows > 1) StoreDotPair(c + ldc, Vector.Sum(s10), Vector.Sum(s11), cols, alpha, beta);
            if (rows > 2) StoreDotPair(c + 2 * ldc, Vector.Sum(s20), Vector.Sum(s21), cols, alpha, beta);
            if (rows > 3) StoreDotPair(c + 3 * ldc, Vector.Sum(s30), Vector.Sum(s31), cols, alpha, beta);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> Fold(Vector512<float> v) => v.GetLower() + v.GetUpper();

        /// <summary>[sum v0, sum v1, sum v2, sum v3] in three horizontal adds.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Reduce4(Vector256<float> v0, Vector256<float> v1, Vector256<float> v2, Vector256<float> v3)
        {
            // hadd(hadd(v0, v1), hadd(v2, v3)) = [v0, v1, v2, v3 summed over lanes 0..3 | over 4..7].
            Vector256<float> u = Avx.HorizontalAdd(Avx.HorizontalAdd(v0, v1), Avx.HorizontalAdd(v2, v3));
            return u.GetLower() + u.GetUpper();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StoreDotRow(float* c, Vector128<float> sums, int cols, float alpha, float beta)
        {
            if (cols == 4)
            {
                Vector128<float> v = sums * alpha;
                if (beta != 0f) v += Vector128.Load(c) * beta;
                v.Store(c);
                return;
            }
            for (int j = 0; j < cols; j++)
            {
                float v = alpha * sums.GetElement(j);
                c[j] = beta == 0f ? v : v + beta * c[j];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void StoreDotPair(float* c, float d0, float d1, int cols, float alpha, float beta)
        {
            float v0 = alpha * d0;
            c[0] = beta == 0f ? v0 : v0 + beta * c[0];
            if (cols > 1)
            {
                float v1 = alpha * d1;
                c[1] = beta == 0f ? v1 : v1 + beta * c[1];
            }
        }

        // ------------------------------------------------------------------------------------
        //  Skinny products (M <= 4): memory bound on B, so read B where it lies.
        // ------------------------------------------------------------------------------------

        /// <summary>B row-major (contiguous along N): C rows += A(i,p) * B[p, :] over column chunks
        /// kept L1-resident, B streamed once per chunk.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SkinnyRowB(int m, int n, int k, float alpha, float* a, long ars, long acs,
            float* b, long brs, float beta, float* c, long ldc)
        {
            const int Chunk = 512;
            int w = Vector<float>.Count;
            for (int j0 = 0; j0 < n; j0 += Chunk)
            {
                int nc = Math.Min(Chunk, n - j0);
                for (int i = 0; i < m; i++)
                {
                    Span<float> row = new Span<float>(c + i * ldc + j0, nc);
                    if (beta == 0f) row.Clear();
                    else if (beta != 1f) System.Numerics.Tensors.TensorPrimitives.Multiply(row, beta, row);
                }

                for (int p = 0; p < k; p++)
                {
                    float* bp = b + p * brs + j0;
                    for (int i = 0; i < m; i++)
                    {
                        float s = alpha * a[i * ars + p * acs];
                        if (s == 0f) continue;
                        float* cp = c + i * ldc + j0;
                        Vector<float> vs = new Vector<float>(s);
                        int j = 0;
                        for (; j + w <= nc; j += w)
                        {
                            Vector<float> acc = Unsafe.ReadUnaligned<Vector<float>>(cp + j);
                            acc += vs * Unsafe.ReadUnaligned<Vector<float>>(bp + j);
                            Unsafe.WriteUnaligned(cp + j, acc);
                        }
                        for (; j < nc; j++) cp[j] += s * bp[j];
                    }
                }
            }
        }

        /// <summary>B given as its transpose (B^T rows contiguous along K) and A rows contiguous:
        /// dot products, two B^T rows at a time against every A row.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void SkinnyDot(int m, int n, int k, float alpha, float* a, long ars,
            float* b, long bcs, float beta, float* c, long ldc)
        {
            int j = 0;
            for (; j + 2 <= n; j += 2)
            {
                float* b0 = b + j * bcs;
                float* b1 = b0 + bcs;
                for (int i = 0; i < m; i++)
                {
                    float* ar = a + i * ars;
                    Dot2(ar, b0, b1, k, out float d0, out float d1);
                    float* cp = c + i * ldc + j;
                    if (beta == 0f)
                    {
                        cp[0] = alpha * d0;
                        cp[1] = alpha * d1;
                    }
                    else
                    {
                        cp[0] = alpha * d0 + beta * cp[0];
                        cp[1] = alpha * d1 + beta * cp[1];
                    }
                }
            }
            for (; j < n; j++)
            {
                float* b0 = b + j * bcs;
                for (int i = 0; i < m; i++)
                {
                    float d0 = System.Numerics.Tensors.TensorPrimitives.Dot(
                        new ReadOnlySpan<float>(a + i * ars, k), new ReadOnlySpan<float>(b0, k));
                    float* cp = c + i * ldc + j;
                    cp[0] = beta == 0f ? alpha * d0 : alpha * d0 + beta * cp[0];
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Dot2(float* a, float* b0, float* b1, int k, out float d0, out float d1)
        {
            int w = Vector<float>.Count;
            Vector<float> s00 = Vector<float>.Zero, s01 = Vector<float>.Zero;
            Vector<float> s10 = Vector<float>.Zero, s11 = Vector<float>.Zero;
            int p = 0;
            for (; p + 2 * w <= k; p += 2 * w)
            {
                Vector<float> x0 = Unsafe.ReadUnaligned<Vector<float>>(a + p);
                Vector<float> x1 = Unsafe.ReadUnaligned<Vector<float>>(a + p + w);
                s00 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p);
                s01 += x1 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p + w);
                s10 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p);
                s11 += x1 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p + w);
            }
            for (; p + w <= k; p += w)
            {
                Vector<float> x0 = Unsafe.ReadUnaligned<Vector<float>>(a + p);
                s00 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b0 + p);
                s10 += x0 * Unsafe.ReadUnaligned<Vector<float>>(b1 + p);
            }
            float r0 = Vector.Sum(s00 + s01);
            float r1 = Vector.Sum(s10 + s11);
            for (; p < k; p++)
            {
                r0 += a[p] * b0[p];
                r1 += a[p] * b1[p];
            }
            d0 = r0;
            d1 = r1;
        }

        // ------------------------------------------------------------------------------------
        //  Per-thread packing buffers (64-byte aligned, grown on demand, kept for the thread's
        //  lifetime: the pool's workers run GEMM after GEMM with the same shapes).
        // ------------------------------------------------------------------------------------

        private static class ThreadBuffers
        {
            [ThreadStatic] private static Holder? _a;
            [ThreadStatic] private static Holder? _b;

            public static float* GetA(long floats) => (_a ??= new Holder()).Get(floats);
            public static float* GetB(long floats) => (_b ??= new Holder()).Get(floats);

            private sealed class Holder
            {
                private float* _ptr;
                private long _capacity;

                public float* Get(long floats)
                {
                    if (floats > _capacity)
                    {
                        // Allocate before freeing: if the allocation throws, the holder still
                        // owns its old buffer (no dangling pointer for the next call or the
                        // finalizer to free again).
                        long cap = Math.Max(floats, _capacity * 2);
                        float* grown = (float*)NativeMemory.AlignedAlloc((nuint)(cap * sizeof(float)), 64);
                        if (_ptr != null) NativeMemory.AlignedFree(_ptr);
                        _ptr = grown;
                        _capacity = cap;
                    }
                    return _ptr;
                }

                ~Holder()
                {
                    if (_ptr != null) NativeMemory.AlignedFree(_ptr);
                }
            }
        }
    }
}
