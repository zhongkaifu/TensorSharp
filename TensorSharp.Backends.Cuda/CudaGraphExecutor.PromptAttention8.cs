// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The 8-bit prompt attention (tensorsharp_flash_prefill's ts_flash_prompt_i8e4_d256 over its band): a prompt chunk's rows
    /// the model marks as one (GraphAttentionRows.PromptChunk) attend with Q K^T on the INT8 tensor cores and P V on the
    /// block-scaled E4M3 ones, where the loaded image carries them (an sm_120a / sm_121a build). Decode rows, verify windows and
    /// drafts never carry the mark, so their arithmetic, and the speculative contract between them, is the 16-bit kernels'.
    /// </summary>
    public sealed partial class CudaGraphExecutor
    {
        /// <summary>Whether marked prompt chunks attend in 8 bits where the image carries the kernels: not yet (the 16-bit
        /// kernels, bit for bit as before), until its quality on long contexts is measured.</summary>
        private const bool PromptAttention8Default = false;

        /// <summary>Tests and the eng probe: null keeps the default; true asks for the 8-bit prompt attention wherever it runs,
        /// false keeps the 16-bit kernel.</summary>
        internal bool? PromptAttention8ForTests { get; set; }

        private bool PromptAttention8Enabled => PromptAttention8ForTests ?? PromptAttention8Default;

        /// <summary>Prompt chunks' attention ops run in 8 bits, and those whose band found no device room (tests, the probe).</summary>
        internal long PromptAttention8Launches { get; private set; }
        internal long PromptAttention8BandRefusals { get; private set; }

        /// <summary>Diagnostic: after each 8-bit launch, the 16-bit kernel over the same rows into scratch, synchronously, and
        /// (launch count, relative RMS of the difference, largest absolute difference) reported.</summary>
        internal Action<long, double, double> PromptAttention8CheckForTests { get; set; }

        /// <summary>Tests: the device memory a band must leave free in place of <see cref="Attn8BandHeadroomBytes"/>, a cached
        /// block of its size not counted as room.</summary>
        internal long? Attn8BandHeadroomForTests { get; set; }

        /// <summary>Device memory a band that is not in the pool's cache must leave free (the direct engine's growth reserve):
        /// the band is the most expendable memory a step asks for, so it never reaches the allocator's out-of-memory reclaim.</summary>
        private const long Attn8BandHeadroomBytes = 512L << 20;

        private static int s_attn8BandWarned;

        /// <summary>
        /// One sequence's marked prompt chunk (<paramref name="nq"/> rows over keys [floor, len) of the head-first F16 caches) on
        /// the 8-bit kernels, or false (nothing launched; the 16-bit kernel runs) unless: the path is on, the heads are 256 wide,
        /// the window holds at least <see cref="CudaFlashPrefillKernels.Attn8MinKeys"/> keys and the chunk at least
        /// <see cref="CudaFlashPrefillKernels.Attn8MinRows"/> rows, nothing records or captures, the image carries the kernels
        /// and the band finds device room without a reclaim.
        /// </summary>
        private bool TryPromptAttention8(Tensor qc, Tensor kc, Tensor vc, int cap, int nq, int len, int floor, int heads, int kvHeads,
            int d, float scale, CudaFlashPrefillKernels flash, out Tensor attended)
        {
            attended = null;
            if (!PromptAttention8Enabled || d != CudaFlashPrefillKernels.Attn8HeadDim || len < CudaFlashPrefillKernels.Attn8MinKeys
                || nq < CudaFlashPrefillKernels.Attn8MinRows || floor < 0 || floor >= len)
                return false;
            var cuda = (CudaAllocator)Allocator;
            // A prompt chunk never records (only decode-size steps do); the check keeps that structural.
            if (CudaDecodeDynParams.IsActive(cuda) || CudaGraphCapture.IsCapturing(cuda) || !flash.Runs8BitPrompt(d))
                return false;
            int bandKeys = CudaFlashPrefillKernels.Attn8BandKeys(len);
            long bandBytes = CudaFlashPrefillKernels.Attn8BandBytes(bandKeys, kvHeads);
            if (!BandHasRoom(cuda, bandBytes))
            {
                PromptAttention8BandRefusals++;
                if (Interlocked.Exchange(ref s_attn8BandWarned, 1) == 0)
                    Console.Error.WriteLine(
                        $"WARNING: [cuda] no device room for the {bandKeys}-key 8-bit prompt-attention band ({bandBytes >> 20} MB beyond a " +
                        $"{(Attn8BandHeadroomForTests ?? Attn8BandHeadroomBytes) >> 20} MB reserve): such prompt chunks attend on the 16-bit " +
                        "kernel instead, slower over long contexts. Reported once.");
                return false;
            }
            int splits = flash.Splits(nq * (heads / kvHeads), kvHeads, len);
            attended = new Tensor(Allocator, DType.Float32, nq, heads, d);
            using var band = new Tensor(Allocator, DType.Float32, bandBytes / sizeof(float));
            using Tensor partial = splits > 1
                ? new Tensor(Allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
            Tensor[] written = partial == null ? new[] { attended, band } : new[] { attended, band, partial };
            Tensor output = attended;
            Launch(new[] { qc, kc, vc }, written, (_, stream) => flash.LaunchPromptAttention8(Ptr(qc), Ptr(kc), Ptr(vc), cap, nq, len, floor,
                heads, kvHeads, scale, splits, Ptr(band), bandKeys, partial == null ? IntPtr.Zero : Ptr(partial), Ptr(output), stream));
            PromptAttention8Launches++;
            if (PromptAttention8CheckForTests is { } check)
                CheckPromptAttention8(check, qc, kc, vc, cap, nq, len, floor, heads, kvHeads, d, scale, splits, flash, attended);
            return true;
        }

        /// <summary>Room for a band of <paramref name="bytes"/>: a cached block of its size, or the driver's free memory past the
        /// band and the reserve. Never the pool's reclaim (a refused band falls back without allocating).</summary>
        private bool BandHasRoom(CudaAllocator cuda, long bytes)
        {
            long headroom = Attn8BandHeadroomForTests ?? Attn8BandHeadroomBytes;
            if (Attn8BandHeadroomForTests == null && cuda.HasCachedBlock(bytes))
                return true;
            long free = cuda.FreeDeviceBytes();
            return free >= bytes && free - bytes >= headroom;
        }

        private void CheckPromptAttention8(Action<long, double, double> check, Tensor qc, Tensor kc, Tensor vc, int cap, int nq, int len,
            int floor, int heads, int kvHeads, int d, float scale, int splits, CudaFlashPrefillKernels flash, Tensor attended)
        {
            var cuda = (CudaAllocator)Allocator;
            using var reference = new Tensor(Allocator, DType.Float32, nq, heads, d);
            using Tensor partial = splits > 1
                ? new Tensor(Allocator, DType.Float32, CudaFlashPrefillKernels.PartialFloats(nq, heads, d, splits)) : null;
            Tensor[] written = partial == null ? new[] { reference } : new[] { reference, partial };
            Launch(new[] { qc, kc, vc }, written, (_, stream) => flash.LaunchAttention(Ptr(qc), Ptr(kc), Ptr(vc), cap, nq, len, floor,
                heads, kvHeads, d, scale, splits, partial == null ? IntPtr.Zero : Ptr(partial), Ptr(reference), stream, prompt: true));
            cuda.Stream.Synchronize();
            int n = checked(nq * heads * d);
            float[] a = attended.GetElementsAsFloat(n), b = reference.GetElementsAsFloat(n);
            double diff = 0, norm = 0, maxAbs = 0;
            for (int i = 0; i < n; i++)
            {
                double e = (double)a[i] - b[i];
                diff += e * e;
                norm += (double)b[i] * b[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(e));
            }
            check(PromptAttention8Launches, norm > 0 ? Math.Sqrt(diff / norm) : Math.Sqrt(diff), maxAbs);
        }
    }
}
