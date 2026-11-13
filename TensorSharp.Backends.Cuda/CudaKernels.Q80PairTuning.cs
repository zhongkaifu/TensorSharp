// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The Q8_0 pair matvec's kernel per call, chosen by measurement on the running GPU. The twin, the lane kernel and
    /// the tensor cores give the same bits (<see cref="LaunchQuantMatmulQ80Pair"/>), so the choice is speed alone and
    /// can never change an output. Their relative speed depends on the GPU: <see cref="Q80PairKernelFor"/> encodes an
    /// H100 NVL measurement, and on an RTX 5090 it missed the fastest at several of Flash Next's decode shapes (attn q
    /// 2560 -> 12288 at 4 rows: twin 22.6 us against its lane 23.8; shexp down 640 -> 2560 at 5-8 rows: lane against its
    /// tensor cores). An engine therefore times every candidate at every weight shape it will launch,
    /// for 1 to <see cref="Q80PairMaxRows"/> rows, once at load (<see cref="TuneQ80Pair"/>), and each launch takes the
    /// winner; a near-tie with the heuristic's pick keeps that pick (<see cref="ChooseQ80PairKernel"/>), so the picks
    /// repeat from run to run. A shape launched without that measurement keeps the heuristic, and says so once.
    /// </summary>
    internal sealed unsafe partial class CudaKernels
    {
        private const int CuDeviceAttributeL2CacheSize = 38;
        /// <summary>Timed graph launches per candidate; the fastest counts (a warm-up launch runs first).</summary>
        private const int Q80PairTimedReps = 5;
        /// <summary>Calls in one timed graph at least (the shape's weights, cycled).</summary>
        private const int Q80PairMinTimedCalls = 32;
        /// <summary>Copies of a shape's weight the timing adds at most to read past the L2.</summary>
        private const int Q80PairMaxColdCopies = 128;

        private static readonly Q80PairKernel[] Q80PairCandidates = { Q80PairKernel.Twin, Q80PairKernel.Lane, Q80PairKernel.Mma };

        /// <summary>Rows a pair kernel takes in one launch: the twin 1-4, the lane kernel 1-8, the tensor cores 1-16.</summary>
        internal static int Q80PairKernelMaxRows(Q80PairKernel kernel)
            => kernel == Q80PairKernel.Twin ? 4 : kernel == Q80PairKernel.Lane ? 8 : Q80PairMaxRows;

        // Per weight shape (inDim, outDim): the kernel for 1..Q80PairMaxRows rows ([rows - 1]), measured on this device.
        // TuneQ80Pair replaces the whole table and never changes one in place, so a launch reads it without a lock.
        private IReadOnlyDictionary<(int InDim, int OutDim), Q80PairKernel[]> q80PairTuned =
            new Dictionary<(int InDim, int OutDim), Q80PairKernel[]>();
        private readonly ConcurrentDictionary<(int InDim, int OutDim), bool> q80PairUntunedWarned = new();

        /// <summary>The kernel <see cref="LaunchQuantMatmulQ80Pair"/> runs for a call on this device.</summary>
        internal Q80PairKernel Q80PairKernelOnDevice(int inDim, int outDim, int rows)
            => PickQ80PairKernel(Volatile.Read(ref q80PairTuned), inDim, outDim, rows, q80PairUntunedWarned, Console.Error);

        /// <summary>
        /// The measured kernel of a tuned shape; for a shape <see cref="TuneQ80Pair"/> did not time, the heuristic
        /// (<see cref="Q80PairKernelFor"/>), with one warning per shape naming it.
        /// </summary>
        internal static Q80PairKernel PickQ80PairKernel(IReadOnlyDictionary<(int InDim, int OutDim), Q80PairKernel[]> tuned,
            int inDim, int outDim, int rows, ConcurrentDictionary<(int InDim, int OutDim), bool> warned, TextWriter log)
        {
            if (tuned.TryGetValue((inDim, outDim), out Q80PairKernel[] picks) && rows >= 1 && rows <= picks.Length)
                return picks[rows - 1];
            if (warned.TryAdd((inDim, outDim), true))
                log.WriteLine($"[cuda] Q8_0 pair {inDim} -> {outDim} was launched without being timed on this GPU: its kernel " +
                    "follows the H100-measured heuristic (CudaKernels.Q80PairKernelFor), which may not be this GPU's fastest.");
            return Q80PairKernelFor(inDim, outDim, rows);
        }

        /// <summary>
        /// Time the pair's kernels on this device at each weight shape in <paramref name="weights"/> (the device
        /// pointers of the Q8_0 weights an engine runs through <see cref="LaunchQuantMatmulQ80Pair"/>), for every row
        /// count it takes, and keep the fastest for each. A candidate runs as one captured graph of calls cycling over
        /// the shape's weights, with copies of one added until they stream past twice the L2, as a decode step reads
        /// them cold; the best of <see cref="Q80PairTimedReps"/> launches counts, and a near-tie keeps the heuristic's
        /// pick (<see cref="ChooseQ80PairKernel"/>). Call it outside any capture, with the
        /// stream idle and VRAM for the copies (they are freed before it returns). Returns the one-line summary an
        /// engine logs: the cost and, per shape, the kernel for each run of rows.
        /// </summary>
        internal string TuneQ80Pair(IReadOnlyList<(IntPtr Weights, int InDim, int OutDim)> weights, IntPtr stream)
        {
            if (!HasQ80Pair || weights == null || weights.Count == 0)
                return null;
            var sw = Stopwatch.StartNew();
            CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int l2, CuDeviceAttributeL2CacheSize, device).ThrowOnError();
            long coldBytes = 2L * Math.Max(l2, 1 << 20);
            var table = new Dictionary<(int InDim, int OutDim), Q80PairKernel[]>(Volatile.Read(ref q80PairTuned));
            var parts = new List<string>();
            foreach (var shape in weights.GroupBy(w => (w.InDim, w.OutDim)))
            {
                var (inDim, outDim) = shape.Key;
                var mats = shape.Select(w => w.Weights).Where(p => p != IntPtr.Zero).Distinct().ToList();
                if (mats.Count == 0 || inDim <= 0 || outDim <= 0 || (inDim & 31) != 0)
                    continue;
                var (picks, copies) = TimeQ80PairShape(mats, inDim, outDim, coldBytes, stream);
                table[(inDim, outDim)] = picks;
                long matBytes = Q80PairMatrixBytes(inDim, outDim);
                long readBytes = matBytes * (mats.Count + copies);
                parts.Add($"{inDim}x{outDim} ({mats.Count} weight{(mats.Count == 1 ? "" : "s")}" +
                    (copies > 0 ? $" + {copies} copies" : "") + $", {readBytes / 1048576.0:F0} MiB" +
                    (readBytes < coldBytes ? ", NOT past the L2 (no VRAM for more copies)" : "") +
                    $"): {DescribeQ80PairPicks(picks)}" + DescribeHeuristicDifference(inDim, outDim, picks));
            }
            Volatile.Write(ref q80PairTuned, table);
            return $"{parts.Count} shape{(parts.Count == 1 ? "" : "s")} x rows 1-{Q80PairMaxRows} timed in " +
                $"{sw.Elapsed.TotalSeconds:F2} s (L2 {l2 / 1048576.0:F0} MiB; a kernel replaces the heuristic's only when " +
                $"faster by more than {Q80PairTieShare * 100:F0}% or {Q80PairTieMicros:F2} us, whichever is larger): {string.Join("; ", parts)}";
        }

        private static long Q80PairMatrixBytes(int inDim, int outDim) => (long)(inDim / 32) * 34 * outDim;

        /// <summary>A measured kernel replaces the heuristic's pick only when faster by more than this share of the
        /// pick's time...</summary>
        internal const double Q80PairTieShare = 0.03;
        /// <summary>...and by more than this many microseconds: below both it is a tie, and the heuristic's pick stays.</summary>
        internal const double Q80PairTieMicros = 0.10;

        /// <summary>
        /// The kernel for one shape and row count from its timings: the fastest, unless it beats the heuristic's pick
        /// by no more than the noise margin (<see cref="Q80PairTieShare"/> of the pick's time or
        /// <see cref="Q80PairTieMicros"/>, whichever is larger). Near-ties then resolve the same way on every run, so
        /// the logged picks are reproducible and a coin flip between equal kernels cannot move a run's speed; a clear
        /// winner still wins.
        /// </summary>
        internal static Q80PairKernel ChooseQ80PairKernel(Q80PairKernel heuristic, IReadOnlyDictionary<Q80PairKernel, double> micros)
        {
            Q80PairKernel best = heuristic;
            double bestUs = double.MaxValue;
            foreach (var (kernel, us) in micros)
                if (us < bestUs)
                {
                    bestUs = us;
                    best = kernel;
                }
            if (!micros.TryGetValue(heuristic, out double heuristicUs))
                return best;
            double margin = Math.Max(Q80PairTieShare * heuristicUs, Q80PairTieMicros);
            return heuristicUs - bestUs > margin ? best : heuristic;
        }

        /// <summary>"twin 1, lane 2-4, mma 5-16": the kernel for each run of row counts.</summary>
        internal static string DescribeQ80PairPicks(Q80PairKernel[] picks)
        {
            var sb = new StringBuilder();
            for (int r = 0; r < picks.Length;)
            {
                int end = r;
                while (end + 1 < picks.Length && picks[end + 1] == picks[r])
                    end++;
                if (sb.Length > 0)
                    sb.Append(", ");
                sb.Append(picks[r].ToString().ToLowerInvariant()).Append(' ').Append(r + 1);
                if (end > r)
                    sb.Append('-').Append(end + 1);
                r = end + 1;
            }
            return sb.ToString();
        }

        private static string DescribeHeuristicDifference(int inDim, int outDim, Q80PairKernel[] picks)
        {
            var rows = new List<int>();
            for (int r = 1; r <= picks.Length; r++)
                if (Q80PairKernelFor(inDim, outDim, r) != picks[r - 1])
                    rows.Add(r);
            return rows.Count == 0 ? " (as the heuristic)" : $" (the heuristic differs at rows {string.Join(",", rows)})";
        }

        private (Q80PairKernel[] Picks, int Copies) TimeQ80PairShape(List<IntPtr> mats, int inDim, int outDim, long coldBytes,
            IntPtr stream)
        {
            long matBytes = Q80PairMatrixBytes(inDim, outDim);
            var scratch = new List<IntPtr>();
            var packedRotation = new List<IntPtr>();
            IntPtr xq = IntPtr.Zero, y = IntPtr.Zero;
            try
            {
                long have = matBytes * mats.Count;
                long want = have >= coldBytes ? 0 : Math.Min(Q80PairMaxColdCopies, (coldBytes - have + matBytes - 1) / matBytes);
                for (long c = 0; c < want; c++)
                {
                    // Out of memory only shortens the rotation; the summary reports a rotation that stays in the L2.
                    if (CudaDriverApi.cuMemAlloc(out IntPtr p, (UIntPtr)matBytes) != 0)
                        break;
                    scratch.Add(p);
                    CudaDriverApi.cuMemcpyDtoDAsync(p, mats[0], (UIntPtr)matBytes, stream).ThrowOnError();
                }
                var rotation = mats.Concat(scratch).ToList();
                // The swapped kernel reads packed copies: enough of them to stream past the L2 as the rest do.
                if (HasQ80PairSwap)
                {
                    long packedBytes = Q80PairPackedBytes(inDim, outDim);
                    long n = Math.Max(1, Math.Min(Q80PairMaxColdCopies, (coldBytes + packedBytes - 1) / packedBytes + 1));
                    for (long c = 0; c < n; c++)
                    {
                        if (CudaDriverApi.cuMemAlloc(out IntPtr p, (UIntPtr)packedBytes) != 0)
                            break;
                        packedRotation.Add(p);
                        LaunchQ80PairPack(rotation[(int)(c % rotation.Count)], p, inDim, outDim, stream);
                    }
                }
                long xqBytes = (long)Q80PairMaxRows * (inDim / 32) * Q81BlockBytes;
                CudaDriverApi.cuMemAlloc(out xq, (UIntPtr)xqBytes).ThrowOnError();
                CudaDriverApi.cuMemsetD8Async(xq, 0, (UIntPtr)xqBytes, stream).ThrowOnError();
                CudaDriverApi.cuMemAlloc(out y, (UIntPtr)((long)Q80PairMaxRows * outDim * sizeof(float))).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();

                int calls = Math.Max(rotation.Count, Q80PairMinTimedCalls);
                var picks = new Q80PairKernel[Q80PairMaxRows];
                var micros = new Dictionary<Q80PairKernel, double>();
                for (int rows = 1; rows <= Q80PairMaxRows; rows++)
                {
                    micros.Clear();
                    foreach (Q80PairKernel kernel in Q80PairCandidates)
                        if (rows <= Q80PairKernelMaxRows(kernel))
                            micros[kernel] = TimeQ80PairKernel(kernel, rotation, xq, y, inDim, outDim, rows, calls, stream);
                    if (packedRotation.Count > 0)
                        micros[Q80PairKernel.Swap] = TimeQ80PairKernel(Q80PairKernel.Swap, packedRotation, xq, y, inDim, outDim, rows,
                            Math.Max(packedRotation.Count, Q80PairMinTimedCalls), stream);
                    picks[rows - 1] = ChooseQ80PairKernel(Q80PairKernelFor(inDim, outDim, rows), micros);
                }
                return (picks, scratch.Count);
            }
            finally
            {
                CudaDriverApi.cuStreamSynchronize(stream);
                foreach (IntPtr p in scratch)
                    CudaDriverApi.cuMemFree(p);
                foreach (IntPtr p in packedRotation)
                    CudaDriverApi.cuMemFree(p);
                if (xq != IntPtr.Zero)
                    CudaDriverApi.cuMemFree(xq);
                if (y != IntPtr.Zero)
                    CudaDriverApi.cuMemFree(y);
            }
        }

        /// <summary>Microseconds a call of <paramref name="kernel"/> takes, the best of <see cref="Q80PairTimedReps"/>
        /// launches of one graph of <paramref name="calls"/> calls cycling over <paramref name="rotation"/>.</summary>
        private double TimeQ80PairKernel(Q80PairKernel kernel, List<IntPtr> rotation, IntPtr xq, IntPtr y, int inDim,
            int outDim, int rows, int calls, IntPtr stream)
        {
            IntPtr graph = IntPtr.Zero, exec = IntPtr.Zero, start = IntPtr.Zero, end = IntPtr.Zero;
            try
            {
                CudaDriverApi.cuStreamBeginCapture(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
                bool captured = false;
                try
                {
                    for (int i = 0; i < calls; i++)
                        if (kernel == Q80PairKernel.Swap)
                            LaunchQuantMatmulQ80PairSwap(rotation[i % rotation.Count], xq, y, inDim, outDim, rows, stream);
                        else
                            LaunchQuantMatmulQ80PairKernel(kernel, rotation[i % rotation.Count], xq, y, inDim, outDim, rows, stream);
                    captured = true;
                }
                finally
                {
                    int rc = CudaDriverApi.cuStreamEndCapture(stream, out graph);
                    if (captured)
                        rc.ThrowOnError();
                }
                CudaDriverApi.cuGraphInstantiateWithFlags(out exec, graph, 0).ThrowOnError();
                CudaDriverApi.cuEventCreate(out start, 0).ThrowOnError();
                CudaDriverApi.cuEventCreate(out end, 0).ThrowOnError();
                CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                double best = double.MaxValue;
                for (int r = 0; r < Q80PairTimedReps; r++)
                {
                    CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                    CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                    CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                    CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                    best = Math.Min(best, ms * 1000.0 / calls);
                }
                return best;
            }
            finally
            {
                if (exec != IntPtr.Zero)
                    CudaDriverApi.cuGraphExecDestroy(exec);
                if (graph != IntPtr.Zero)
                    CudaDriverApi.cuGraphDestroy(graph);
                if (start != IntPtr.Zero)
                    CudaDriverApi.cuEventDestroy(start);
                if (end != IntPtr.Zero)
                    CudaDriverApi.cuEventDestroy(end);
            }
        }
    }
}
