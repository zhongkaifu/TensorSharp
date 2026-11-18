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
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using TensorSharp.Cpu;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // Multi-row quantized GEMM for the pure-C# CPU backend.
    //
    // The per-row path (TryAddmmQuantizedToFloat32's DotQuantized loop) calls
    // one weight-row x activation-row dot per output element, so every weight
    // block is re-decoded (nibble unpack, 6-bit scale unpack, fp16 scales) for
    // EVERY activation row. At prefill widths that decode is most of the work:
    // DiffusionGemma's dense MLP measured ~36 GFLOPS that way.
    //
    // Here a pair of weight rows ("columns" of the output) is decoded ONCE into
    // an L1-resident scratch - unsigned int8 values in natural element order,
    // pre-broadcast int16 sub-block scales, float block scales - and then run
    // against every activation row of the tile with a register-blocked
    // microkernel: 8 rows x 2 columns in one zmm (AVX-512BW) or 4 rows x 1
    // column (AVX2). The integer arithmetic is the one the per-row kernels use
    // (vpmaddubsw into int16, vpmaddwd against the sub-block scale into int32),
    // so the per-sub-block integer sums are exact and only the float scaling is
    // re-associated.
    //
    // Activations are quantized exactly as before (same Q8_K / Q8_0 values and
    // scales); only their layout differs: all int8 values of a row first
    // (32-byte aligned chunks, so no load splits a cache line), then the side
    // data the kernels want pre-computed once per row instead of once per
    // (row, column) pair - float scales, and for Q4_K/Q5_K the d8 * bsum
    // products that carry the K-quant min term.
    //
    // TS_CPU_DISABLE_AVX512=1 runs every AVX-512 path of ManagedQuantizedOps
    // (GEMM kernels, quantizer, per-row Q4_0/Q8_0 dots, MaxAbs) in its AVX2
    // form, so the AVX2 path can be tested on an AVX-512 machine; the ISA flags
    // come from TensorSharp.Cpu.CpuIsa, shared with every other managed CPU
    // kernel. Hosts without AVX2+FMA (and ARM64) keep the per-row path.
    //
    // These kernels are reached by every caller of the managed matmul, not only
    // the pure-C# backend: the DSV4 CUDA executor's host-MoE offload
    // (TryAddmmQuantizedBatch) and the CUDA/MLX "no device kernel" fallback
    // (ModelBase.AddmmQuantManaged) run them too. That is intended - they are
    // the same managed code, faster, and within ~3e-6 of the per-row path - but
    // the float panel keeps dequantizing through NativeDequant there, as
    // DequantMatMulColumns did (see ManagedQuantGemm.Float.cs).
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        /// <summary>Kernel selection for the multi-row GEMM. <see cref="Auto"/>
        /// honours TS_CPU_DISABLE_AVX512; the others exist so tests and benchmarks can
        /// compare paths in one process. <see cref="PerRow"/> is the per-row dot path
        /// that hosts without AVX2+FMA (ARM64 included) run.</summary>
        internal enum QGemmIsa
        {
            Auto = 0,
            PerRow = 1,
            Avx2 = 2,
            Avx512 = 3,
        }

        private enum QGemmFamily
        {
            None,
            /// <summary>Q4_K / Q5_K: unsigned values, 32-element sub-blocks with a
            /// 6-bit scale and a 6-bit min (min term via Q8_K bsums).</summary>
            KMin,
            /// <summary>Q6_K: unsigned 0..63 values, 16-element int8 scales, -32
            /// zero point folded in as an exact integer correction off the bsums.</summary>
            KOfs,
            /// <summary>Q4_0 / Q5_0: unsigned values with a zero point, folded in
            /// through the activation block sums.</summary>
            Q0Unsigned,
            /// <summary>Q8_0: signed values (sign moved onto the activation).</summary>
            Q0Signed,
        }

        // Smallest row count that takes the GEMM: one, for every type. Each
        // output of the GEMM goes through the same operations whatever the
        // height of the tile it lands in, so a row's result is independent of
        // how many rows share the call - decode vs speculative verify vs
        // continuous batching, a dense call vs an MoE batch job - only if single
        // rows take the GEMM too; the per-row dot sums in another order. It
        // costs nothing measurable at M = 1: DRAM-bound matvecs (CpuQuantBench
        // gemm decode, weights rotated past the L3) ran 1.0-2.1x the per-row
        // speed for the K-quants, 1.1-1.5x for Q4_0, 1.0-1.7x for Q5_0,
        // 1.0-1.1x for BF16 and 0.93-1.10x for Q8_0; cache-resident MoE-sized
        // rows 1.1-1.6x for Q4_0 and 0.6-1.5x for Q8_0 (8 of 10 samples above
        // 1). Single runs on the 8-core laptop that measured this vary ~15%.
        //
        // TS_CPU_QGEMM_MIN_ROWS=N (diagnostic) sends calls and batch jobs with
        // fewer rows to the per-row path, which gives up that invariance.
        private static readonly int QGemmMinRowsOverride = (int)Math.Clamp(EnvLong("TS_CPU_QGEMM_MIN_ROWS", 0), 0, int.MaxValue);

        private static int QGemmMinRows(GgmlTensorType type)
            => QGemmMinRowsOverride > 0 ? QGemmMinRowsOverride : 1;

        /// <summary>True when every row count of every GEMM type (and of the
        /// float panel's) takes the GEMM - the condition for the batch-size
        /// invariance above. A TS_CPU_QGEMM_MIN_ROWS above one or a host without AVX2
        /// gives it up.</summary>
        internal static bool QGemmRoutingIsBatchInvariant =>
            QGemmMinRowsOverride <= 1 && ResolveQGemmIsa(QGemmIsa.Auto) != QGemmIsa.PerRow;
        // Minimum multiply-accumulates per parallel task. ~1M MACs is ~50 us on
        // one core here - big enough to bury the pool dispatch, small enough that
        // an MoE expert (4 rows x 2816 x 1408 = 16M MACs) still fans out.
        private static readonly long QGemmMinTaskMacs = EnvLong("TS_CPU_QGEMM_TASK_MACS", 1L << 20);
        // Activation bytes one row block may hold. The block is re-read once per
        // decoded column pair, so it has to stay in L2 next to the pair scratch
        // (~18 KB at K=4096); 1.25 MB L2 parts keep ~40% of it for this.
        private static readonly long QGemmRowBlockBytes = EnvLong("TS_CPU_QGEMM_L2_BYTES", 512 * 1024);

        // Diagnostic: TS_CPU_QGEMM_VERIFY=1 re-runs every GEMM through the per-row
        // path and reports the largest relative difference seen so far (stderr),
        // which checks the kernels on a real model's weights and activations.
        private static readonly bool QGemmVerify =
            Environment.GetEnvironmentVariable("TS_CPU_QGEMM_VERIFY") == "1";
        private static double _qgemmVerifyWorst;
        private static readonly object QGemmVerifyLock = new object();

        internal static bool QGemmAvx2Supported => CpuIsa.HasAvx2Fma;

        internal static bool QGemmAvx512Supported => QGemmAvx2Supported && CpuIsa.HasAvx512;

        /// <summary>Concrete kernel set for a request: PerRow when the ISA is missing.</summary>
        internal static QGemmIsa ResolveQGemmIsa(QGemmIsa requested)
            => ResolveQGemmIsa(requested, CpuIsa.Avx512DisabledByEnv, QGemmAvx2Supported, QGemmAvx512Supported);

        /// <summary>The routing rule itself, with the switches and the host's ISA
        /// as arguments (tests check it without touching process state).
        /// Explicit Avx2/Avx512 requests ignore the switches.</summary>
        internal static QGemmIsa ResolveQGemmIsa(
            QGemmIsa requested, bool avx512Disabled, bool avx2Supported, bool avx512Supported)
        {
            switch (requested)
            {
                case QGemmIsa.PerRow:
                    return QGemmIsa.PerRow;
                case QGemmIsa.Avx512:
                    return avx512Supported ? QGemmIsa.Avx512 : avx2Supported ? QGemmIsa.Avx2 : QGemmIsa.PerRow;
                case QGemmIsa.Avx2:
                    return avx2Supported ? QGemmIsa.Avx2 : QGemmIsa.PerRow;
                default:
                    if (!avx2Supported) return QGemmIsa.PerRow;
                    return avx512Supported && !avx512Disabled ? QGemmIsa.Avx512 : QGemmIsa.Avx2;
            }
        }

        private static QGemmFamily GetQGemmFamily(GgmlTensorType type, int inDim)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_K:
                case GgmlTensorType.Q5_K:
                    return inDim % QK_K == 0 ? QGemmFamily.KMin : QGemmFamily.None;
                case GgmlTensorType.Q6_K:
                    return inDim % QK_K == 0 ? QGemmFamily.KOfs : QGemmFamily.None;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q5_0:
                    return inDim % QK8_0 == 0 ? QGemmFamily.Q0Unsigned : QGemmFamily.None;
                case GgmlTensorType.Q8_0:
                    return inDim % QK8_0 == 0 ? QGemmFamily.Q0Signed : QGemmFamily.None;
                default:
                    return QGemmFamily.None;
            }
        }

        private static bool IsKFamily(QGemmFamily family) => family == QGemmFamily.KMin || family == QGemmFamily.KOfs;

        // GEMM activation row: int8 values [inDim] | per-block side data, padded
        // to 64 bytes so every row (and every 32-value chunk) stays aligned.
        //   K family : aux[nsb] x 32 B (KMin: 8 floats d8*bsum-pair, KOfs: 16 int16 bsums) | d8[nsb] floats
        //   Q0 family: dx[nb] floats (fp16-rounded block scale) | sxAdj[nb] int32 (zero point * block sum / 2)
        private static int QGemmActRowBytes(QGemmFamily family, int inDim)
        {
            long bytes = IsKFamily(family)
                ? inDim + (long)(inDim / QK_K) * 36
                : inDim + (long)(inDim / QK8_0) * 8;
            return checked((int)((bytes + 63) & ~63L));
        }

        /// <summary>Scratch bytes one decoded column pair needs.</summary>
        private static int QGemmPairScratchBytes(QGemmFamily family, int inDim)
            => IsKFamily(family) ? inDim / QK_K * QKSbStride : Q0PairScratchBytes(inDim / QK8_0);

        // Per-thread, 64-byte aligned decode scratch. Pinned on the POH so the
        // address stays valid for the lifetime of the array, and reclaimed with
        // the thread (the pool's workers live for the process).
        [ThreadStatic] private static byte[] _qgemmScratch;

        private static unsafe byte* QGemmScratch(int bytes)
        {
            byte[] buf = _qgemmScratch;
            if (buf == null || buf.Length < bytes + 64)
            {
                buf = GC.AllocateUninitializedArray<byte>(Math.Max(bytes + 64, 64 * 1024), pinned: true);
                _qgemmScratch = buf;
            }
            byte* p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buf));
            return (byte*)(((nint)p + 63) & ~(nint)63);
        }

        /// <summary>
        /// Multi-row GEMM entry: <c>output[r, c] = dot(weightRow(c), input[r])</c>
        /// for the types the tiled kernels cover. Returns false (and writes
        /// nothing) when the type/shape/ISA is not handled, so the caller can run
        /// the per-row path.
        /// </summary>
        internal static unsafe bool TryQGemm(
            int ggmlType,
            IntPtr weights,
            int inDim,
            int outDim,
            float* input,
            int inputRowStride,
            int rowCount,
            float* output,
            int outputRowStride,
            ParallelOptions options,
            QGemmIsa isa)
        {
            var type = (GgmlTensorType)ggmlType;
            QGemmFamily family = GetQGemmFamily(type, inDim);
            if (family == QGemmFamily.None || rowCount <= 0 || outDim <= 0)
                return false;
            isa = ResolveQGemmIsa(isa);
            if (isa == QGemmIsa.PerRow)
                return false;

            int actStride = QGemmActRowBytes(family, inDim);
            long actBytes = (long)rowCount * actStride + 64;
            if (actBytes > int.MaxValue)
                return false;

            int dop = ResolveDop(options);
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)actBytes);
            try
            {
                fixed (byte* rentedBase = rented)
                {
                    byte* act = (byte*)(((nint)rentedBase + 63) & ~(nint)63);
                    bool avx512 = isa == QGemmIsa.Avx512;
                    QuantizeRowsQGemm(type, family, input, inputRowStride, rowCount, inDim, act, actStride, options, dop, avx512);

                    int rowBytes = (int)RowSize(ggmlType, inDim);
                    QGemmPlan plan = PlanQGemm(rowCount, inDim, outDim, rowBytes, actStride, dop);
                    nint w = weights, a = (nint)act, o = (nint)output;
                    int outStride = outputRowStride;
                    void RunTask(int t)
                    {
                        int cb = t % plan.ColBlocks;
                        int rb = t / plan.ColBlocks;
                        int col0 = cb * plan.PairsPerBlock * 2;
                        int col1 = Math.Min(outDim, col0 + plan.PairsPerBlock * 2);
                        int row0 = rb * plan.RowsPerBlock;
                        int row1 = Math.Min(rowCount, row0 + plan.RowsPerBlock);
                        byte* scratch = QGemmScratch(QGemmPairScratchBytes(family, inDim));
                        QGemmTile(type, family, avx512, (byte*)w, rowBytes, inDim, outDim,
                            (byte*)a, actStride, (float*)o, outStride, col0, col1, row0, row1, scratch);
                    }

                    int tasks = plan.RowBlocks * plan.ColBlocks;
                    if (tasks > 1 && dop > 1)
                        RunParallelBlocks(tasks, options, RunTask);
                    else
                        for (int t = 0; t < tasks; t++) RunTask(t);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
            return true;
        }

        /// <summary>TS_CPU_QGEMM_VERIFY: compare a finished GEMM with the per-row path.</summary>
        private static unsafe void VerifyQGemmAgainstPerRow(
            string what, int ggmlType, IntPtr weights, int inDim, int outDim, float* input, int inputRowStride,
            int rowCount, float* output, int outputRowStride, bool floatPanel)
        {
            float[] reference = new float[(long)rowCount * outDim];
            fixed (float* r = reference)
            {
                if (floatPanel)
                {
                    long rowBytes = RowSize(ggmlType, inDim);
                    float[] scratch = new float[inDim];
                    fixed (float* s = scratch)
                        DequantMatMulColumns(ggmlType, (byte*)weights, rowBytes, inDim, outDim,
                            input, inputRowStride, rowCount, r, outDim, 0, outDim, s);
                }
                else
                {
                    TryAddmmQuantizedToFloat32(ggmlType, weights, inDim, outDim, input, inputRowStride, rowCount,
                        r, outDim, null, QGemmIsa.PerRow);
                }
            }
            double maxRef = 1e-30, maxDiff = 0;
            for (int row = 0; row < rowCount; row++)
                for (int c = 0; c < outDim; c++)
                {
                    double e = reference[(long)row * outDim + c];
                    double a = output[(long)row * outputRowStride + c];
                    maxRef = Math.Max(maxRef, Math.Abs(e));
                    maxDiff = Math.Max(maxDiff, double.IsNaN(a) || double.IsNaN(e) ? double.PositiveInfinity : Math.Abs(a - e));
                }
            double rel = maxDiff / maxRef;
            lock (QGemmVerifyLock)
            {
                if (rel > _qgemmVerifyWorst || rel > 1e-4)
                {
                    _qgemmVerifyWorst = Math.Max(_qgemmVerifyWorst, rel);
                    Console.Error.WriteLine(
                        $"[qgemm-verify] {what} {(GgmlTensorType)ggmlType} M={rowCount} K={inDim} N={outDim}: " +
                        $"max|gemm-perRow|/max|perRow| = {rel:E2} (worst so far {_qgemmVerifyWorst:E2})");
                }
            }
        }

        private readonly struct QGemmPlan
        {
            public readonly int RowsPerBlock;
            public readonly int RowBlocks;
            public readonly int PairsPerBlock;
            public readonly int ColBlocks;

            public QGemmPlan(int rowsPerBlock, int rowBlocks, int pairsPerBlock, int colBlocks)
            {
                RowsPerBlock = rowsPerBlock;
                RowBlocks = rowBlocks;
                PairsPerBlock = pairsPerBlock;
                ColBlocks = colBlocks;
            }
        }

        /// <summary>
        /// How many tasks a matmul is worth: enough arithmetic per task to bury
        /// the dispatch, or - for the few-row, bandwidth-bound shapes - enough
        /// weight bytes per task by the same rule as ParallelColumnBlock; never
        /// more than a few per worker.
        /// </summary>
        private static long QGemmWantTasks(long macs, long weightBytes, long rows, int dop)
        {
            if (dop <= 1) return 1;
            long perTaskBytes = Math.Max(MinBytesPerParallelTaskFloor, MinBytesPerParallelTask / Math.Clamp(rows, 1, 8));
            long byWork = Math.Max(macs / Math.Max(1, QGemmMinTaskMacs), weightBytes / Math.Max(1, perTaskBytes));
            return Math.Clamp(byWork, 1, (long)dop * TasksPerWorker);
        }

        /// <summary>
        /// Tile the (rows x column pairs) space. Rows are cut only when the
        /// activation block would fall out of L2 (every decoded pair re-reads
        /// it); columns are cut so there are enough tasks for the pool, sized by
        /// work rather than by thread count (see ParallelColumnBlock).
        /// </summary>
        private static QGemmPlan PlanQGemm(int rowCount, int inDim, int outDim, long weightRowBytes, int actStride, int dop)
        {
            int pairs = (outDim + 1) / 2;
            // Full-size row blocks with a short remainder block LAST: the pool
            // hands tasks out in index order, so the remainder's small tasks
            // fill the gaps the big ones leave (largest-first scheduling). An
            // even split (M = 256, K = 4096: 88/88/80 rows instead of
            // 112/112/32) measured 1-5% slower.
            int rowsPerBlock = rowCount;
            long blockRows = QGemmRowBlockBytes / Math.Max(1, actStride);
            if (blockRows < rowCount)
                rowsPerBlock = (int)Math.Max(8, blockRows & ~7L);
            int rowBlocks = (rowCount + rowsPerBlock - 1) / rowsPerBlock;

            long wantTasks = QGemmWantTasks((long)rowCount * outDim * inDim, weightRowBytes * outDim, rowCount, dop);
            long colBlocks = Math.Clamp((wantTasks + rowBlocks - 1) / rowBlocks, 1, pairs);
            int pairsPerBlock = (int)((pairs + colBlocks - 1) / colBlocks);
            colBlocks = (pairs + pairsPerBlock - 1) / pairsPerBlock;
            return new QGemmPlan(rowsPerBlock, rowBlocks, pairsPerBlock, (int)colBlocks);
        }

        /// <summary>One task: decode each column pair of [col0, col1) once, then
        /// run it against every row tile of [row0, row1).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QGemmTile(
            GgmlTensorType type, QGemmFamily family, bool avx512,
            byte* weights, int rowBytes, int inDim, int outDim,
            byte* act, int actStride, float* output, int outStride,
            int col0, int col1, int row0, int row1, byte* scratch)
        {
            int tileRows = avx512 ? 8 : 4;
            for (int c = col0; c < col1; c += 2)
            {
                bool second = c + 1 < outDim;
                byte* wa = weights + (long)c * rowBytes;
                DecodeQGemmPair(type, wa, second ? wa + rowBytes : wa, inDim, scratch);
                for (int r = row0; r < row1; r += tileRows)
                {
                    int rows = Math.Min(tileRows, row1 - r);
                    RunQGemmKernel(family, avx512, rows, scratch, act + (long)r * actStride, actStride, inDim,
                        output + (long)r * outStride + c, outStride, second);
                }
            }
        }

        // ------------------------------------------------------------------
        // Batched (MoE) GEMM: every job's column pairs in one flat task space.
        // ------------------------------------------------------------------

        private static unsafe bool TryQGemmBatch(
            GgmlTensorType type,
            int inDim,
            int inputRowStride,
            ReadOnlySpan<QuantMatMulJob> jobs,
            ParallelOptions options,
            QGemmIsa isa)
        {
            QGemmFamily family = GetQGemmFamily(type, inDim);
            if (family == QGemmFamily.None)
                return false;
            isa = ResolveQGemmIsa(isa);
            if (isa == QGemmIsa.PerRow)
                return false;

            // --- quantize each DISTINCT input block once (same rule as the per-row batch) ---
            int n = jobs.Length;
            var actOf = new int[n];
            var blockInput = new IntPtr[n];
            var blockRows = new int[n];
            var blockOffset = new long[n];
            int nBlocks = 0;
            long totalRows = 0;
            for (int j = 0; j < n; j++)
            {
                int found = -1;
                for (int b = 0; b < nBlocks; b++)
                    if (blockInput[b] == jobs[j].Input && blockRows[b] == jobs[j].RowCount)
                    {
                        found = b;
                        break;
                    }
                if (found < 0)
                {
                    found = nBlocks++;
                    blockInput[found] = jobs[j].Input;
                    blockRows[found] = jobs[j].RowCount;
                    blockOffset[found] = totalRows;
                    totalRows += jobs[j].RowCount;
                }
                actOf[j] = found;
            }

            int actStride = QGemmActRowBytes(family, inDim);
            long actBytes = totalRows * actStride + 64;
            if (actBytes > int.MaxValue)
                return false;

            int dop = ResolveDop(options);
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)Math.Max(64, actBytes));
            try
            {
                fixed (byte* rentedBase = rented)
                {
                    byte* act = (byte*)(((nint)rentedBase + 63) & ~(nint)63);
                    bool avx512 = isa == QGemmIsa.Avx512;
                    if (dop > 1 && totalRows >= 8 && totalRows * inDim >= 64 * 1024)
                    {
                        // A prefill-sized MoE batch gathers thousands of rows; quantize
                        // them in parallel, flattened across the distinct input blocks.
                        // That is a second dispatch (the compute needs every row
                        // quantized first), which on the spinning pool costs
                        // microseconds against milliseconds of serial quantization
                        // (1055 routed rows x 2816 took 1.6 ms, CpuQuantBench batch).
                        // Decode-sized batches stay under the threshold and keep
                        // one dispatch.
                        var rowSrc = new nint[totalRows];
                        for (int b = 0; b < nBlocks; b++)
                            for (int row = 0; row < blockRows[b]; row++)
                                rowSrc[blockOffset[b] + row] = blockInput[b] + (nint)((long)row * inputRowStride * sizeof(float));
                        int rowsPerTask = (int)Math.Max(4, (totalRows + dop * TasksPerWorker - 1) / (dop * TasksPerWorker));
                        int quantTasks = (int)((totalRows + rowsPerTask - 1) / rowsPerTask);
                        nint dstBase = (nint)act;
                        int width = inDim;
                        RunParallelBlocks(quantTasks, options, t =>
                        {
                            long r1 = Math.Min(rowSrc.Length, (long)(t + 1) * rowsPerTask);
                            for (long r = (long)t * rowsPerTask; r < r1; r++)
                                QuantizeRowQGemm(type, family, (float*)rowSrc[r], (byte*)dstBase + r * actStride, width, avx512);
                        });
                    }
                    else
                    {
                        for (int b = 0; b < nBlocks; b++)
                        {
                            float* src = (float*)blockInput[b];
                            byte* dst = act + blockOffset[b] * actStride;
                            for (int row = 0; row < blockRows[b]; row++)
                                QuantizeRowQGemm(type, family, src + (long)row * inputRowStride, dst + (long)row * actStride, inDim, avx512);
                        }
                    }

                    // Pairs per task from the total work, as in PlanQGemm.
                    int rowBytes = (int)RowSize((int)type, inDim);
                    long macs = 0, totalPairs = 0, totalCols = 0;
                    for (int j = 0; j < n; j++)
                    {
                        if (jobs[j].RowCount <= 0) continue;
                        macs += (long)jobs[j].RowCount * jobs[j].OutDim * inDim;
                        totalPairs += (jobs[j].OutDim + 1) / 2;
                        totalCols += jobs[j].OutDim;
                    }
                    // Rows per weight byte = the jobs' average row count.
                    long avgRows = macs / Math.Max(1, totalCols * inDim);
                    long wantTasks = QGemmWantTasks(macs, totalCols * rowBytes, avgRows, dop);
                    int pairsPerTask = (int)Math.Max(1, (totalPairs + wantTasks - 1) / wantTasks);

                    var taskStart = new int[n + 1];
                    int totalTasks = 0;
                    for (int j = 0; j < n; j++)
                    {
                        taskStart[j] = totalTasks;
                        int jobPairs = (jobs[j].OutDim + 1) / 2;
                        if (jobs[j].RowCount > 0)
                            totalTasks += (jobPairs + pairsPerTask - 1) / pairsPerTask;
                    }
                    taskStart[n] = totalTasks;

                    // Span cannot cross the lambda, so copy the jobs' raw fields.
                    var wPtr = new nint[n];
                    var oPtr = new nint[n];
                    var outDims = new int[n];
                    var rowCnt = new int[n];
                    var outStrides = new int[n];
                    var actOff = new long[n];
                    for (int j = 0; j < n; j++)
                    {
                        wPtr[j] = jobs[j].Weights;
                        oPtr[j] = jobs[j].Output;
                        outDims[j] = jobs[j].OutDim;
                        rowCnt[j] = jobs[j].RowCount;
                        outStrides[j] = jobs[j].OutputRowStride;
                        actOff[j] = blockOffset[actOf[j]] * actStride;
                    }
                    nint actAddr = (nint)act;
                    int scratchBytes = QGemmPairScratchBytes(family, inDim);

                    void RunTask(int t)
                    {
                        int j = 0;
                        while (j + 1 < n && taskStart[j + 1] <= t) j++;
                        int local = t - taskStart[j];
                        int col0 = local * pairsPerTask * 2;
                        int col1 = Math.Min(outDims[j], col0 + pairsPerTask * 2);
                        byte* scratch = QGemmScratch(scratchBytes);
                        QGemmTile(type, family, avx512, (byte*)wPtr[j], rowBytes, inDim, outDims[j],
                            (byte*)actAddr + actOff[j], actStride, (float*)oPtr[j], outStrides[j],
                            col0, col1, 0, rowCnt[j], scratch);
                    }

                    if (totalTasks > 1 && dop > 1)
                        RunParallelBlocks(totalTasks, options, RunTask);
                    else
                        for (int t = 0; t < totalTasks; t++) RunTask(t);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
            return true;
        }

        // ------------------------------------------------------------------
        // Activation quantization into the GEMM layout.
        // ------------------------------------------------------------------

        private static unsafe void QuantizeRowsQGemm(
            GgmlTensorType type, QGemmFamily family, float* input, int inputRowStride, int rowCount, int inDim,
            byte* act, int actStride, ParallelOptions options, int dop, bool avx512)
        {
            // Rows are independent; hand them out in chunks sized by elements so a
            // 4096-row prefill is a few dozen tasks, not thousands.
            long elements = (long)rowCount * inDim;
            if (dop > 1 && rowCount >= 8 && elements >= 64 * 1024)
            {
                int rowsPerTask = (int)Math.Max(4, Math.Min(rowCount, (32 * 1024 + inDim - 1) / inDim));
                int tasks = Math.Min((rowCount + rowsPerTask - 1) / rowsPerTask, dop * TasksPerWorker);
                rowsPerTask = (rowCount + tasks - 1) / tasks;
                nint src = (nint)input, dst = (nint)act;
                RunParallelBlocks(tasks, options, t =>
                {
                    int r0 = t * rowsPerTask, r1 = Math.Min(rowCount, r0 + rowsPerTask);
                    for (int r = r0; r < r1; r++)
                        QuantizeRowQGemm(type, family, (float*)src + (long)r * inputRowStride,
                            (byte*)dst + (long)r * actStride, inDim, avx512);
                });
                return;
            }
            for (int r = 0; r < rowCount; r++)
                QuantizeRowQGemm(type, family, input + (long)r * inputRowStride, act + (long)r * actStride, inDim, avx512);
        }

        /// <summary>
        /// Quantize one activation row into the GEMM layout. The int8 values and
        /// scales are bit-identical to QuantizeF32ToQ8_K / QuantizeF32ToQ8_0
        /// (same max-abs scale, same round-half-to-even, same +-127 clamp).
        /// <paramref name="avx512"/> is the kernel set the GEMM runs, so an AVX2
        /// GEMM (TS_CPU_DISABLE_AVX512=1, or an explicit Avx2 request) quantizes
        /// with AVX2 as well.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void QuantizeRowQGemm(GgmlTensorType type, QGemmFamily family, float* src, byte* dst, int inDim, bool avx512)
        {
            sbyte* qs = (sbyte*)dst;
            if (IsKFamily(family))
            {
                int nsb = inDim / QK_K;
                byte* aux = dst + inDim;
                float* d8 = (float*)(aux + nsb * 32);
                int* groupSums = stackalloc int[QK_K / 16];
                for (int sb = 0; sb < nsb; sb++)
                {
                    float* x = src + sb * QK_K;
                    float maxAbs = MaxAbs(x, QK_K);
                    float scale = maxAbs / 127.0f;
                    d8[sb] = scale;
                    sbyte* q = qs + sb * QK_K;
                    byte* a = aux + sb * 32;
                    if (scale == 0.0f)
                    {
                        Unsafe.InitBlockUnaligned(q, 0, QK_K);
                        Unsafe.InitBlockUnaligned(a, 0, 32);
                        continue;
                    }

                    QuantizeInt8Groups16Simd(x, 1.0f / scale, q, QK_K, groupSums, avx512);
                    if (family == QGemmFamily.KMin)
                    {
                        float* bsF = (float*)a;
                        for (int j = 0; j < 8; j++)
                            bsF[j] = scale * (groupSums[2 * j] + groupSums[2 * j + 1]);
                    }
                    else
                    {
                        short* bs = (short*)a;
                        for (int g = 0; g < 16; g++)
                            bs[g] = (short)groupSums[g];
                    }
                }
                return;
            }

            int nb = inDim / QK8_0;
            float* dx = (float*)(dst + inDim);
            int* sxAdj = (int*)(dst + inDim + nb * 4);
            // Q4_0's zero point is 8 and Q5_0's 16. The kernels fold it in as
            // an integer correction: their four-block reduction leaves two
            // int32 lanes per (column, block), so each lane gets half of
            // zeroPoint * sum(x); the single-block tail (eight lanes) a quarter
            // of that.
            int zeroShare = type == GgmlTensorType.Q5_0 ? 8 : 4;
            int* groupSums2 = stackalloc int[2];
            for (int b = 0; b < nb; b++)
            {
                float* x = src + b * QK8_0;
                float maxAbs = MaxAbs(x, QK8_0);
                float scale = maxAbs / 127.0f;
                // The per-row kernels read the block scale back from its fp16
                // encoding, so the GEMM uses that same rounded value.
                dx[b] = (float)(System.Half)scale;
                sbyte* q = qs + b * QK8_0;
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(q, 0, QK8_0);
                    sxAdj[b] = 0;
                    continue;
                }

                QuantizeInt8Groups16Simd(x, 1.0f / scale, q, QK8_0, groupSums2, avx512);
                sxAdj[b] = zeroShare * (groupSums2[0] + groupSums2[1]);
            }
        }

        /// <summary>
        /// <c>q[i] = clamp(round_half_even(x[i] * invScale), -127, 127)</c> for
        /// <paramref name="n"/> (a multiple of 16) values, plus the sum of each
        /// 16-value group, for the per-row quantizers (QuantizeF32ToQ8_0/_K, which
        /// the Q8_0 KV-cache writer shares). Bit-identical to the scalar
        /// <c>ClampToInt8(MathF.Round(x * invScale))</c> for every input, NaN
        /// and infinities included (see the SIMD variants).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void QuantizeInt8Groups16(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            if (CpuIsa.Avx512)
                QuantizeInt8Groups16Avx512(x, invScale, q, n, groupSums);
            else if (Avx2.IsSupported)
                QuantizeInt8Groups16Avx2(x, invScale, q, n, groupSums);
            else if (AdvSimd.Arm64.IsSupported)
                QuantizeInt8Groups16Neon(x, invScale, q, n, groupSums);
            else
                QuantizeInt8Groups16Scalar(x, invScale, q, n, groupSums);
        }

        /// <summary>The GEMM's quantizer: the SIMD variant of its kernel set. The GEMM
        /// itself only runs where AVX2 is present, but the layout is also built through
        /// the test hook below on hosts without it (ARM64, DOTNET_EnableAVX2=0), which
        /// take the scalar loop - every variant produces the same bits.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void QuantizeInt8Groups16Simd(float* x, float invScale, sbyte* q, int n, int* groupSums, bool avx512)
        {
            if (avx512)
                QuantizeInt8Groups16Avx512(x, invScale, q, n, groupSums);
            else if (Avx2.IsSupported)
                QuantizeInt8Groups16Avx2(x, invScale, q, n, groupSums);
            else if (AdvSimd.Arm64.IsSupported)
                QuantizeInt8Groups16Neon(x, invScale, q, n, groupSums);
            else
                QuantizeInt8Groups16Scalar(x, invScale, q, n, groupSums);
        }

        // The scalar reference converts with C#'s (int) cast, which saturates:
        // NaN -> 0, +-Inf and anything past the int range -> int.MaxValue /
        // int.MinValue, then clamped to +-127. vcvtps2dq returns int.MinValue for
        // all of those (-127 after an integer clamp), which differs for NaN and
        // +Inf. Both occur only in degenerate blocks - a NaN activation, or a
        // block whose max|x| is below ~3.7e-37 so that 1/scale overflows to +Inf
        // (and 0 * Inf is NaN) - but vectorizing must not change what the
        // per-row path computes. So NaN lanes are zeroed and the clamp is done in
        // float BEFORE the conversion: +-127 are integers and rounding is
        // monotonic, so round(clamp(v)) == clamp(round(v)) for every finite v.

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void QuantizeInt8Groups16Avx512(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            Vector512<float> inv = Vector512.Create(invScale);
            Vector512<float> lo = Vector512.Create(-127f), hi = Vector512.Create(127f);
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                Vector512<float> v = Avx512F.Multiply(Avx512F.LoadVector512(x + i), inv);
                v &= Vector512.Equals(v, v);                        // NaN -> +0
                v = Avx512F.Min(Avx512F.Max(v, lo), hi);
                Vector512<int> r = Avx512F.ConvertToVector512Int32(Avx512F.RoundScale(v, 0));
                Avx512F.ConvertToVector128SByte(r).Store(q + i);
                groupSums[g] = Vector512.Sum(r);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void QuantizeInt8Groups16Avx2(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            Vector256<float> inv = Vector256.Create(invScale);
            Vector256<float> lo = Vector256.Create(-127f), hi = Vector256.Create(127f);
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                Vector256<float> v0 = Avx.Multiply(Avx.LoadVector256(x + i), inv);
                Vector256<float> v1 = Avx.Multiply(Avx.LoadVector256(x + i + 8), inv);
                v0 = Avx.And(v0, Avx.CompareOrdered(v0, v0));      // NaN -> +0
                v1 = Avx.And(v1, Avx.CompareOrdered(v1, v1));
                v0 = Avx.Min(Avx.Max(v0, lo), hi);
                v1 = Avx.Min(Avx.Max(v1, lo), hi);
                Vector256<int> r0 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(v0));
                Vector256<int> r1 = Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(v1));
                // int32 -> int16 -> int8 packs work per 128-bit lane:
                // [r0.lo r1.lo | r0.hi r1.hi] -> bytes need a dword permute.
                Vector256<short> s = Avx2.PackSignedSaturate(r0, r1);
                Vector256<sbyte> b = Avx2.PackSignedSaturate(s, s);
                b = Avx2.PermuteVar8x32(b.AsInt32(), Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7)).AsSByte();
                b.GetLower().Store(q + i);
                groupSums[g] = Vector256.Sum(Avx2.Add(r0, r1));
            }
        }

        internal static unsafe void QuantizeInt8Groups16Scalar(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                int sum = 0;
                for (int k = 0; k < 16; k++)
                {
                    sbyte v = ClampToInt8(MathF.Round(x[i + k] * invScale));
                    q[i + k] = v;
                    sum += v;
                }
                groupSums[g] = sum;
            }
        }

        // ---- test hooks ----------------------------------------------------

        /// <summary>GEMM activation row size for <paramref name="type"/>, or 0 when
        /// the type has no multi-row kernel.</summary>
        internal static int QGemmActivationRowBytes(GgmlTensorType type, int inDim)
        {
            QGemmFamily family = GetQGemmFamily(type, inDim);
            return family == QGemmFamily.None ? 0 : QGemmActRowBytes(family, inDim);
        }

        /// <summary>Quantize one row into the GEMM layout (see QuantizeRowQGemm), with the
        /// AVX-512 quantizer when <paramref name="avx512"/> and the host has it, else AVX2,
        /// else scalar.</summary>
        internal static unsafe void QGemmQuantizeActivationRow(GgmlTensorType type, float* src, byte* dst, int inDim,
            bool avx512 = true)
            => QuantizeRowQGemm(type, GetQGemmFamily(type, inDim), src, dst, inDim, avx512 && QGemmAvx512Supported);
    }
}
