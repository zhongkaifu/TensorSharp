// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;
using TensorSharp.Graph;

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
        /// <summary>The prompt attention a prompt chunk's rows run unless the operator chooses another
        /// (<see cref="PromptAttentionConfig"/>: <c>--prompt-attention</c>, <c>TS_PROMPT_ATTENTION</c>): the 8-bit path where it
        /// runs, the 16-bit-run kernels (<see cref="PromptAttentionConfig.Fallback"/>) elsewhere. The user chose int8 as the
        /// default (issue #383, 2026-11-20), accepting the pre-registered quality gates (design t2-attn8 sec. 12.3) it failed on
        /// Qwen3.8-27B NVFP4 (RTX PRO 4500), on a few long C# code windows (2026-11-18): with q8 activations on the prompt rows
        /// of both arms, the prompt rows moved +0.8% PPL at 64K keys and +18% on a 96K window; as served (W4A4 prompt rows), the
        /// decode rows after a code prompt moved +0.8% to +6.4% PPL at 8K-64K keys and +94% on the 96K window, where ggml_cuda as
        /// served moved them +1.5% at 8K, -0.4% at 64K and -22% at 96K. Prose and long-context retrieval did not move. On that
        /// card a fresh 47K-69K prompt then prefills 18-22% faster than on the 16-bit-run kernels, as fast as the direct
        /// engine of 038fb303. <c>--prompt-attention f16</c> restores the 16-bit-run kernels, bit for bit.</summary>
        private const PromptAttentionPath PromptAttentionDefault = PromptAttentionConfig.Default;

        /// <summary>The operator's choice when the executor was made (<see cref="PromptAttentionConfig.FromEnvironment"/>; null:
        /// nothing chosen; an unknown value refuses the executor rather than run another path).</summary>
        private readonly PromptAttentionPath? _promptAttentionConfigured = PromptAttentionConfig.FromEnvironment();

        /// <summary>Tests and the eng probe: a path in place of the configured one (null keeps it).</summary>
        internal PromptAttentionPath? PromptAttentionForTests { get; set; }

        /// <summary>The path chosen for this executor (the seam, else the operator's), or null when nothing was.</summary>
        private PromptAttentionPath? PromptAttentionChosen => PromptAttentionForTests ?? _promptAttentionConfigured;

        /// <summary>The prompt attention this executor's prompt chunks ask for: the chosen path, else the default (int8, which
        /// runs f16 where the 8-bit path cannot).</summary>
        internal PromptAttentionPath PromptAttention => PromptAttentionChosen ?? PromptAttentionDefault;

        /// <summary>Tests and the eng probe: null keeps the chosen path; true asks for the 8-bit prompt attention wherever it
        /// runs, false keeps a 16-bit kernel.</summary>
        internal bool? PromptAttention8ForTests { get; set; }

        private bool PromptAttention8Enabled => PromptAttention8ForTests ?? PromptAttention == PromptAttentionPath.Int8;

        /// <summary>The 8-bit path was asked for explicitly (<c>--prompt-attention int8</c> or a seam), so a chunk it cannot take
        /// is reported; as the default, its fallback to f16 is the default too and says nothing.</summary>
        private bool PromptAttention8Explicit => PromptAttention8ForTests == true || PromptAttentionChosen == PromptAttentionPath.Int8;

        /// <summary>Prompt chunks' attention ops on the kernels with F32 accumulation (tests, the probe).</summary>
        internal long PromptAttentionF32AccLaunches { get; private set; }

        private static int s_promptAttentionNoted;
        private static int s_attn8DeclinedWarned;

        /// <summary>Once a process: which prompt attention the first prompt chunk on a path other than f16 took and what it runs
        /// on this device.</summary>
        private void NotePromptAttentionOnce(PromptAttentionPath path, CudaFlashPrefillKernels flash)
        {
            if (Interlocked.Exchange(ref s_promptAttentionNoted, 1) != 0) return;
            string detail = path == PromptAttentionPath.F32Acc
                ? $"F32 accumulation runs at {flash.F32AccumulationRate:F2} of the F16-accumulation tensor rate here" +
                  (flash.PromptAccumulatesInF32 ? "" : ", so it may be slower than f16 on this device")
                : "8-bit Q K^T and E4M3 P V for 256-wide heads over windows of 1,024 keys or more";
            string source = PromptAttentionChosen == null
                ? "the default where it runs"
                : $"{PromptAttentionConfig.Flag} / {PromptAttentionConfig.EnvironmentVariable}";
            Console.Error.WriteLine(
                $"[cuda] prompt attention: {PromptAttentionConfig.Name(path)} ({source}; {detail}). It moves a few long C# code " +
                $"windows (issue #376's quality gates); {PromptAttentionConfig.Flag} f16 keeps the 16-bit kernels. Decode rows, " +
                "verify windows and drafts keep the 16-bit kernels.");
        }

        /// <summary>Once a process: the 8-bit prompt attention was chosen explicitly but cannot run here, so prompt chunks attend
        /// on the 16-bit kernel. Nothing for the default: there f16 is the default.</summary>
        private void WarnAttn8DeclinedOnce(string why)
        {
            if (PromptAttention8Explicit && Interlocked.Exchange(ref s_attn8DeclinedWarned, 1) == 0)
                Console.Error.WriteLine(
                    $"WARNING: [cuda] {PromptAttentionConfig.Flag} int8 was chosen, but {why}: prompt chunks attend on the 16-bit " +
                    "kernel (f16) instead. Reported once.");
        }

        /// <summary>Prompt chunks' attention ops run in 8 bits, and those whose band found no device room (tests, the probe).</summary>
        internal long PromptAttention8Launches { get; private set; }
        internal long PromptAttention8BandRefusals { get; private set; }

        /// <summary>Diagnostic: after each 8-bit launch, the 16-bit kernel over the same rows into scratch, synchronously, and
        /// (launch count, relative RMS of the difference, largest absolute difference) reported.</summary>
        internal Action<long, double, double> PromptAttention8CheckForTests { get; set; }

        /// <summary>Tests: the device memory a band must leave free in place of <see cref="Attn8BandHeadroomBytes"/>, a cached
        /// block of its size not counted as room; the pool's other idle large blocks still make room as they do without the
        /// seam (<see cref="BandTrimBytes"/>), so a test can make the band give them back, and long.MaxValue refuses every band
        /// without giving any back.</summary>
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
            if (!PromptAttention8Enabled) return false;
            if (d != CudaFlashPrefillKernels.Attn8HeadDim)
            {
                WarnAttn8DeclinedOnce($"the model's attention heads are {d} wide (the 8-bit kernels take {CudaFlashPrefillKernels.Attn8HeadDim})");
                return false;
            }
            // Short windows and chunks keep the 16-bit kernel by design: the band costs more than it saves there.
            if (len < CudaFlashPrefillKernels.Attn8MinKeys || nq < CudaFlashPrefillKernels.Attn8MinRows || floor < 0 || floor >= len)
                return false;
            var cuda = (CudaAllocator)Allocator;
            // A prompt chunk never records (only decode-size steps do); the check keeps that structural.
            if (CudaDecodeDynParams.IsActive(cuda) || CudaGraphCapture.IsCapturing(cuda))
                return false;
            if (!flash.Runs8BitPrompt(d))
            {
                WarnAttn8DeclinedOnce("the loaded CUDA image does not carry the 8-bit kernels (an sm_120a or sm_121a build does)");
                return false;
            }
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
            NotePromptAttentionOnce(PromptAttentionPath.Int8, flash);
            if (PromptAttention8CheckForTests is { } check)
                CheckPromptAttention8(check, qc, kc, vc, cap, nq, len, floor, heads, kvHeads, d, scale, splits, flash, attended);
            return true;
        }

        /// <summary>Room for a band of <paramref name="bytes"/>: a cached block of its size, or the driver's free memory past the
        /// band and the reserve once the pool has given back as many of its idle large blocks as that needs
        /// (<see cref="BandTrimBytes"/>). A server near the device's edge keeps its room there: the slots and retained
        /// conversations the model released for a long prompt (its EnsureDeviceRoom counts the pool's cached blocks as free)
        /// wait in the pool's cache, so the driver alone had no room for even a 32 MB band beside a 47K-token prompt of
        /// Qwen3.8-27B with DFlash2 on a 32 GB card. Never the out-of-memory ladder (no GC, no drain): a band that still has no
        /// room falls back without allocating.</summary>
        private bool BandHasRoom(CudaAllocator cuda, long bytes)
        {
            bool seam = Attn8BandHeadroomForTests != null;
            long headroom = Attn8BandHeadroomForTests ?? Attn8BandHeadroomBytes;
            long trim = BandTrimBytes(!seam && cuda.HasCachedBlock(bytes), cuda.FreeDeviceBytes(), cuda.ReclaimableBytes, bytes, headroom);
            if (trim <= 0)
                return trim == 0;
            cuda.TrimLargeCache(trim, keepBlockBytes: cuda.AllocationBlockBytes(bytes));
            long free = cuda.FreeDeviceBytes();
            return free >= bytes && free - bytes >= headroom;
        }

        /// <summary>
        /// What a band of <paramref name="bytes"/> needs from the pool's idle large blocks (<paramref name="reclaimable"/>, none
        /// of the band's own size when <paramref name="cachedOfItsSize"/> is false): 0 when it has room as things are (a cached
        /// block of its size, or <paramref name="driverFree"/> past the band and <paramref name="headroom"/>), the bytes to give
        /// back when those blocks make the room, -1 when even all of them would not.
        /// </summary>
        internal static long BandTrimBytes(bool cachedOfItsSize, long driverFree, long reclaimable, long bytes, long headroom)
        {
            if (cachedOfItsSize || (driverFree >= bytes && driverFree - bytes >= headroom))
                return 0;
            long shortfall = headroom > long.MaxValue - bytes - Math.Max(0, -driverFree) ? long.MaxValue : bytes + headroom - driverFree;
            return reclaimable >= shortfall ? shortfall : -1;
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
