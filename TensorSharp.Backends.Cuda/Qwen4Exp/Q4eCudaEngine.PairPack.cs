// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Packed copies of the Q8_0 dense projections for the pair's swapped tensor-core kernel (CudaKernels.Q80PairKernel.Swap).
// The GGUF's 34-byte Q8_0 blocks are 2-byte aligned, so the pair's kernels read them with 16-bit loads and keep little in
// flight: at 8 rows a call reads HBM at 0.4-1.1 TB/s and the dense projections took as long as the routed experts in a
// batched decode step (nsys, 16 streams on two H100 NVL). The packed copy holds each 16-column tile's tensor-core fragments
// as aligned 16-byte words; the swapped kernel computes the same bits from it (same int32 block dots, the same fma chains
// and trees), 1.1-2.0x faster than the best of the other three at 8 rows. The copy costs the weight's bytes again
// (Q80PairPackedBytes), counted into the layers' placement; the tuning (TuneQ80Pair) then picks it per shape and row count
// where it is faster. TS_Q80_PAIR_SWAP=0 packs nothing.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        /// <summary>Bytes of packed copies on each device (tests, the load log).</summary>
        internal long[] PairPackedBytes { get; private set; }

        /// <summary>Whether a weight of this descriptor gets a packed copy: a dense Q8_0 matrix of whole blocks the pair
        /// takes (not a stacked expert, not a decode-once weight), with the swapped kernel wanted.</summary>
        private static bool PacksForPair(in CudaWeightDesc w)
            => SwapWanted && Q80Pair && w.IsValid && w.GgmlType == TQ8_0 && !w.DecodeOnce && Math.Max(1, w.Ne2) == 1
               && (w.Ne0 & 31) == 0 && w.Ne0 > 0 && w.Ne1 > 0;

        private static readonly bool SwapWanted = Environment.GetEnvironmentVariable("TS_Q80_PAIR_SWAP") != "0";

        /// <summary>The packed copies a layer's dense weights add to its placement.</summary>
        private static long PairPackedBytesOf(System.Collections.Generic.IEnumerable<CudaWeightDesc> weights)
        {
            long n = 0;
            foreach (var w in weights)
                if (PacksForPair(w))
                    n += CudaWeightArena.Align(CudaKernels.Q80PairPackedBytes(w.Ne0, w.Ne1));
            return n;
        }

        /// <summary>Pack every dense weight the pair takes on its device (after the upload, before the tuning).</summary>
        private void PackPairWeights()
        {
            PairPackedBytes = new long[_devs.Length];
            if (!SwapWanted || !Q80Pair)
                return;
            foreach (DevLayer L in _layers)
            {
                Dev dev = _devs[L.Device];
                Pack(dev, ref L.HcAttnDown); Pack(dev, ref L.HcAttnUp); Pack(dev, ref L.HcFfnDown); Pack(dev, ref L.HcFfnUp);
                Pack(dev, ref L.Qkv); Pack(dev, ref L.Gate); Pack(dev, ref L.SsmOut);
                Pack(dev, ref L.Wq); Pack(dev, ref L.Wk); Pack(dev, ref L.Wv); Pack(dev, ref L.Wo);
                Pack(dev, ref L.PleKey); Pack(dev, ref L.PleValue);
                Pack(dev, ref L.GateShexp); Pack(dev, ref L.UpShexp); Pack(dev, ref L.DownShexp);
            }
            Dev last = _devs[_lastDev];
            Pack(last, ref _output); Pack(last, ref _outHcDown); Pack(last, ref _outHcUp);
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            _devs[0].MakeCurrent();
            long total = 0;
            foreach (long b in PairPackedBytes)
                total += b;
            if (total > 0)
                Console.Error.WriteLine($"[q4e-cuda] Q8_0 pair: packed copies of the dense projections for the swapped tensor-core "
                    + $"kernel, {string.Join(" + ", Array.ConvertAll(PairPackedBytes, b => (b / 1048576.0).ToString("F0")))} MiB (TS_Q80_PAIR_SWAP=0: none)");
        }

        private void Pack(Dev dev, ref DeviceWeight w)
        {
            if (w.Ptr == IntPtr.Zero || w.PairPacked != IntPtr.Zero || !TakesQ80Pair(w, 1, dev.Alloc.Kernels.HasQ80Pair)
                || !dev.Alloc.Kernels.HasQ80PairSwap)
                return;
            dev.MakeCurrent();
            long bytes = CudaKernels.Q80PairPackedBytes(w.Ne0, w.Ne1);
            Tensor t;
            try { t = AllocT(dev, DType.UInt8, bytes); }
            catch (Exception ex) when (ex is OutOfMemoryException or CudaException or InvalidOperationException)
            {
                Console.Error.WriteLine($"[q4e-cuda] Q8_0 pair: no VRAM for a packed copy of a {w.Ne0} -> {w.Ne1} weight on GPU {dev.Ordinal} "
                    + $"({ex.Message}); it keeps the GGUF layout's kernels.");
                return;
            }
            dev.Alloc.Kernels.LaunchQ80PairPack(w.Ptr, Ptr(t), w.Ne0, w.Ne1, dev.Stream);
            w.PairPacked = Ptr(t);
            PairPackedBytes[dev.Ordinal] += bytes;
        }
    }
}
