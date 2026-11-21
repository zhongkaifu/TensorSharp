// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the tensor-core flash prefill attention kernels
// (native/kernels/tensorsharp_flash_prefill.cu, compiled to tensorsharp_flash_prefill.ptx): the
// attention CudaGraphExecutor runs prompt chunks, verify windows and decode rows on, over a head-first F16 or q8_0 cache.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaFlashPrefillKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_flash_prefill.ptx";
        public const string CapsKernelName = "ts_flash_prefill_caps";

        /// <summary>TS_FLASH_PREFILL_VERSION in tensorsharp_flash_prefill.cu: bump both together whenever a
        /// kernel is added or changes its signature (CudaPrefillModuleTests pins that they agree).</summary>
        public const int CurrentVersion = 14;

        /// <summary>Query rows (query heads x positions of one KV head) one block takes, and keys per tile.</summary>
        public const int BlockRows = 128, TileKeys = 32;

        private const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
        private readonly IntPtr _attn128, _attn256, _rows128, _rows256, _window128, _window256, _combine, _windowCombine, _listedCombine;
        private readonly IntPtr _prompt128, _prompt256;
        // The prompt kernels with F32 accumulation (ts_flash_prompt2_*) and their tail combine.
        private readonly IntPtr _prompt2_128, _prompt2_256, _prompt2Q8128, _prompt2Q8256, _prompt2Combine;
        private readonly IntPtr _band256, _attn8_256;
        private readonly int _smem128, _smem256;
        // The same kernels over a q8_0 cache (ts_flash_*_q8_d*), and their shared memory (raw rows and one F16 tile pair).
        private readonly IntPtr _attnQ8128, _attnQ8256, _rowsQ8128, _rowsQ8256, _windowQ8128, _windowQ8256, _promptQ8128, _promptQ8256;
        private readonly int _smemQ8128, _smemQ8256;

        /// <summary>The 8-bit prompt attention's head width, and the fewest keys a chunk attends before it takes it (the direct
        /// engine's Attn8MinKeys: below them the band is most of the work).</summary>
        public const int Attn8HeadDim = 256, Attn8MinKeys = 1024;

        /// <summary>The fewest prompt rows the 8-bit attention takes: below them the band (converted once per layer whatever
        /// the rows) costs more than the rows save. Measured on an RTX PRO 4500 (sm_120, eng/PromptAttentionProbe's row sweep
        /// over 8K-64K keys and groups of 4 and 6): from 128 rows the 8-bit pair with its band beats the 16-bit kernel in every
        /// cell (1.19x at the worst), at 64 rows it loses past 32K keys (0.91x). sm_121 (GB10) takes sm_120's value until it is
        /// measured there. Fixed per architecture, never chosen by timing at run time.</summary>
        public const int Attn8MinRows = 128;

        /// <summary>The band is sized in quanta of this many keys (the direct engine's Attn8BandStepKeys), so the pool's cached
        /// block of one quantum serves every layer and chunk whose keys it holds.</summary>
        public const int Attn8BandStepKeys = 16384;

        /// <summary>Keys of the band for a window of <paramref name="len"/> keys: <see cref="Attn8BandStepKeys"/> quanta.</summary>
        public static int Attn8BandKeys(int len) => (int)(((long)len + Attn8BandStepKeys - 1) / Attn8BandStepKeys * Attn8BandStepKeys);

        /// <summary>Byte offsets of a band's sections (K8, K scales, V^T, V scales; each 256-byte aligned) and its size, for
        /// <paramref name="bandKeys"/> keys of <paramref name="kvHeads"/> KV heads (ts_flash_prompt_band_f16_d256's layout).</summary>
        public static (long K8, long KScales, long VT, long VScales, long Bytes) Attn8BandLayout(int bandKeys, int kvHeads)
        {
            static long Align(long bytes) => (bytes + 255) / 256 * 256;
            long keys = (long)bandKeys * kvHeads;
            long k8 = 0, ks = Align(keys * Attn8HeadDim), vt = ks + Align(keys * sizeof(float)), vs = vt + Align(keys * Attn8HeadDim);
            return (k8, ks, vt, vs, vs + Align(keys / TileKeys));
        }

        /// <summary>Bytes of a band of <paramref name="bandKeys"/> keys for <paramref name="kvHeads"/> KV heads: about
        /// 516 bytes a key and KV head.</summary>
        public static long Attn8BandBytes(int bandKeys, int kvHeads) => Attn8BandLayout(bandKeys, kvHeads).Bytes;

        /// <summary>Dynamic shared memory of ts_flash_prompt_i8e4_d256 (its TsFpA8Smem): two stages of a 32-key K8 tile (272-byte
        /// rows), its V^T tile (256 rows of 48 bytes) and its 32 K scales.</summary>
        public const int Attn8SharedBytes = 2 * TileKeys * (Attn8HeadDim + 16) + 2 * Attn8HeadDim * (TileKeys + 16) + 2 * TileKeys * sizeof(float);

        private CudaFlashPrefillKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
            if (!IsCurrent || !ArchGuardedKernelsUsable) return;
            _attn128 = module.GetFunction("ts_flash_prefill_f16_d128");
            _attn256 = module.GetFunction("ts_flash_prefill_f16_d256");
            _rows128 = module.GetFunction("ts_flash_decode_rows_f16_d128");
            _rows256 = module.GetFunction("ts_flash_decode_rows_f16_d256");
            _window128 = module.GetFunction("ts_flash_decode_window_f16_d128");
            _window256 = module.GetFunction("ts_flash_decode_window_f16_d256");
            _combine = module.GetFunction("ts_flash_prefill_combine_f32");
            _windowCombine = module.GetFunction("ts_flash_window_combine_f32");
            _listedCombine = module.GetFunction("ts_flash_listed_combine_f32");
            _prompt128 = module.GetFunction("ts_flash_prompt_f16_d128");
            _prompt256 = module.GetFunction("ts_flash_prompt_f16_d256");
            (_smem128, _smem256) = (SharedBytes(128), SharedBytes(256));
            CudaDriverApi.cuFuncSetAttribute(_attn128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_attn256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem256).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_rows128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_rows256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem256).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_window128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_window256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem256).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_prompt128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_prompt256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem256).ThrowOnError();
            _attnQ8128 = module.GetFunction("ts_flash_prefill_q8_d128");
            _attnQ8256 = module.GetFunction("ts_flash_prefill_q8_d256");
            _rowsQ8128 = module.GetFunction("ts_flash_decode_rows_q8_d128");
            _rowsQ8256 = module.GetFunction("ts_flash_decode_rows_q8_d256");
            _windowQ8128 = module.GetFunction("ts_flash_decode_window_q8_d128");
            _windowQ8256 = module.GetFunction("ts_flash_decode_window_q8_d256");
            _promptQ8128 = module.GetFunction("ts_flash_prompt_q8_d128");
            _promptQ8256 = module.GetFunction("ts_flash_prompt_q8_d256");
            (_smemQ8128, _smemQ8256) = (SharedBytes(128, q8: true), SharedBytes(256, q8: true));
            foreach (IntPtr fn in new[] { _attnQ8128, _rowsQ8128, _windowQ8128, _promptQ8128 })
                CudaDriverApi.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smemQ8128).ThrowOnError();
            foreach (IntPtr fn in new[] { _attnQ8256, _rowsQ8256, _windowQ8256, _promptQ8256 })
                CudaDriverApi.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smemQ8256).ThrowOnError();
            _prompt2_128 = module.GetFunction("ts_flash_prompt2_f16_d128");
            _prompt2_256 = module.GetFunction("ts_flash_prompt2_f16_d256");
            _prompt2Q8128 = module.GetFunction("ts_flash_prompt2_q8_d128");
            _prompt2Q8256 = module.GetFunction("ts_flash_prompt2_q8_d256");
            _prompt2Combine = module.GetFunction("ts_flash_prompt2_combine_f32");
            CudaDriverApi.cuFuncSetAttribute(_prompt2_128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_prompt2_256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smem256).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_prompt2Q8128, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smemQ8128).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_prompt2Q8256, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _smemQ8256).ThrowOnError();
            CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device).ThrowOnError();
            MultiprocessorCount = sms;
            F32AccumulationRate = MeasureF32AccumulationRate(module, sms);
            PromptAccumulatesInF32 = F32AccumulationRate >= 0.75;
            // The 8-bit prompt attention: only an sm_120a / sm_121a image carries its block-scaled E4M3 MMA.
            Attn8Code = ReadAttn8Caps(module);
            if (Attn8Code)
            {
                _band256 = module.GetFunction("ts_flash_prompt_band_f16_d256");
                _attn8_256 = module.GetFunction("ts_flash_prompt_i8e4_d256");
            }
        }

        public int MultiprocessorCount { get; }

        /// <summary>Whether a prompt chunk's 16-bit attention takes <see cref="LaunchPromptAttention"/> (F32 accumulation) rather
        /// than <see cref="LaunchAttention"/>'s 16-bit-run prompt kernels: where the tensor cores accumulate in F32 at no less than
        /// three quarters of their F16-accumulation rate (<see cref="F32AccumulationRate"/>). Workstation and datacenter parts run
        /// both at one rate (RTX PRO 4500 Blackwell: 1.00); GeForce parts run F32 accumulation at half rate, where the F32
        /// kernels' tensor time alone would exceed the 16-bit-run kernels' whole time.</summary>
        public bool PromptAccumulatesInF32 { get; }

        /// <summary>The tensor cores' F16 rate with F32 accumulation over their rate with F16 accumulation, timed once when the
        /// module loads (ts_flash_prefill_mma_rate, the faster of three runs each, alternating); 0 where it cannot run: a module
        /// from before the probe, or a probe that fails (no device room for its 4-byte sink, a failed event or launch). The
        /// probe only describes the device, so its failure never refuses the module (<see cref="CudaPrefillModule"/>'s
        /// TryCreate refuses one whose constructor throws: every flash kernel would go with it).</summary>
        public double F32AccumulationRate { get; }

        /// <summary>Tests: the result code of the rate probe's sink allocation in place of cuMemAlloc's (for example 2,
        /// CUDA_ERROR_OUT_OF_MEMORY: a full device), for every module loaded while it is set.</summary>
        internal static Func<int> RateProbeSinkAllocationForTests { get; set; }

        /// <summary><see cref="F32AccumulationRate"/> of the current context's device. Requires the module's context to be
        /// current; synchronous (about a millisecond).</summary>
        private static unsafe double MeasureF32AccumulationRate(CudaModule module, int sms)
        {
            IntPtr fn;
            try { fn = module.GetFunction("ts_flash_prefill_mma_rate"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { return 0; } // CUDA_ERROR_NOT_FOUND: a module from before it
            IntPtr sink = IntPtr.Zero, start = IntPtr.Zero, end = IntPtr.Zero;
            try
            {
                (RateProbeSinkAllocationForTests?.Invoke() ?? CudaDriverApi.cuMemAlloc(out sink, new UIntPtr(sizeof(float)))).ThrowOnError();
                CudaDriverApi.cuEventCreate(out start, 0).ThrowOnError();
                CudaDriverApi.cuEventCreate(out end, 0).ThrowOnError();
                float best16 = float.MaxValue, best32 = float.MaxValue;
                int a0 = 0, a1 = 1024;
                IntPtr a2 = sink;
                void** args = stackalloc void*[] { &a0, &a1, &a2 };
                for (int run = 0; run < 4; run++)
                    for (int f32 = 0; f32 < 2; f32++)
                    {
                        a0 = f32;
                        CudaDriverApi.cuEventRecord(start, IntPtr.Zero).ThrowOnError();
                        CudaDriverApi.cuLaunchKernel(fn, (uint)(4 * sms), 1, 1, 128, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                        CudaDriverApi.cuEventRecord(end, IntPtr.Zero).ThrowOnError();
                        CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                        CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                        if (run == 0) continue;   // the first of each warms the clocks
                        if (f32 == 0) best16 = Math.Min(best16, ms); else best32 = Math.Min(best32, ms);
                    }
                return best32 > 0 ? best16 / best32 : 0;
            }
            catch (CudaException)
            {
                return 0;
            }
            finally
            {
                if (start != IntPtr.Zero) CudaDriverApi.cuEventDestroy(start);
                if (end != IntPtr.Zero) CudaDriverApi.cuEventDestroy(end);
                if (sink != IntPtr.Zero) CudaDriverApi.cuMemFree(sink);
            }
        }

        /// <summary>Whether the loaded image carries the 8-bit prompt attention (ts_flash_prefill_attn8_caps: an sm_120a /
        /// sm_121a image). False on the committed baseline, a plain compute_12x PTX and every other architecture.</summary>
        public bool Attn8Code { get; }

        /// <summary>Whether <see cref="LaunchPromptAttention8"/> runs here for <paramref name="headDim"/>-wide heads.</summary>
        public bool Runs8BitPrompt(int headDim) => IsCurrent && ArchGuardedKernelsUsable && Attn8Code && headDim == Attn8HeadDim;

        /// <summary>The attn8 probe of <paramref name="module"/> (one launch; false when the image has no such kernel, a module
        /// from before it). Requires the module's context to be current; synchronous.</summary>
        internal static unsafe bool ReadAttn8Caps(CudaModule module)
        {
            IntPtr caps;
            try { caps = module.GetFunction("ts_flash_prefill_attn8_caps"); }
            catch (CudaException ex) when (ex.ErrorCode == 500) { return false; } // CUDA_ERROR_NOT_FOUND
            CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(sizeof(int))).ThrowOnError();
            try
            {
                int host = -1;
                CudaDriverApi.cuMemcpyHtoD(probe, (IntPtr)(&host), new UIntPtr(sizeof(int))).ThrowOnError();
                IntPtr p = probe;
                void** args = stackalloc void*[] { &p };
                CudaDriverApi.cuLaunchKernel(caps, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)(&host), probe, new UIntPtr(sizeof(int))).ThrowOnError();
                return host == 1;
            }
            finally
            {
                CudaDriverApi.cuMemFree(probe);
            }
        }

        /// <summary>Dynamic shared memory of the <paramref name="headDim"/> kernel (its TsFpAttnSmem): two stages of K and
        /// V tiles, <see cref="TileKeys"/> rows of headDim + 8 halves each; over a q8_0 cache (TsFpAttnSmemQ8)
        /// <see cref="Q8Stages"/> stages of K and V tiles of raw rows (<see cref="Q8RowBytes"/> each) and one F16 tile pair their
        /// values are dequantized into.</summary>
        public static int SharedBytes(int headDim, bool q8 = false)
            => q8 ? Q8Stages * 2 * TileKeys * Q8RowBytes(headDim) + 2 * TileKeys * (headDim + 8) * 2 : 2 * 2 * TileKeys * (headDim + 8) * 2;

        /// <summary>TS_FP_Q8_STAGES: raw q8_0 tiles the _q8 kernels keep in flight (a tile is half an F16 tile's bytes).</summary>
        public const int Q8Stages = 3;

        /// <summary>Bytes of a q8_0 cache row of <paramref name="headDim"/> values (ggml's block_q8_0: an F16 scale and 32 int8
        /// codes, 34 bytes, per 32 values).</summary>
        public static int Q8RowBytes(int headDim) => headDim / 32 * 34;

        /// <summary>Whether <see cref="LaunchAttention"/> runs here for <paramref name="headDim"/>-wide heads (sm_80 code).</summary>
        public bool RunsAttention(int headDim) => IsCurrent && ArchGuardedKernelsUsable && (headDim == 128 || headDim == 256);

        /// <summary>Whether the launchers take a q8_0 cache (q8: true) for <paramref name="headDim"/>-wide heads: the _q8 kernels
        /// of a current image (version 12 on) with its sm_80 code.</summary>
        public bool RunsAttentionQ8(int headDim) => RunsAttention(headDim);

        /// <summary>Key splits for <paramref name="rows"/> query rows of each of <paramref name="kvHeads"/> KV heads over
        /// <paramref name="keys"/> keys: enough blocks for about two waves, each split at least four key tiles.</summary>
        public int Splits(int rows, int kvHeads, int keys)
        {
            int blocks = (rows + BlockRows - 1) / BlockRows * kvHeads, splits = 1;
            while (blocks * splits * 2 <= 2 * MultiprocessorCount && keys / (splits * 2) >= 4 * TileKeys && splits < 64)
                splits *= 2;
            return splits;
        }

        /// <summary>The keys <see cref="Splits"/> takes for a step of <paramref name="length"/> keys over a padded window of
        /// <paramref name="windowKeys"/>: the window of <paramref name="bucket"/>-key steps the length falls in, at most the
        /// window (a window padded to the slot's capacity splits each length as the bucket window around it does).</summary>
        public static int BucketKeys(int windowKeys, int length, int bucket)
            => bucket <= 0 ? windowKeys : (int)Math.Min(windowKeys, ((long)length + bucket - 1) / bucket * bucket);

        /// <summary>Floats of split partials <paramref name="splits"/> splits write for nq queries of nHead heads.</summary>
        public static long PartialFloats(int nq, int nHead, int headDim, int splits) => (long)nq * nHead * splits * (headDim + 2);

        /// <summary>
        /// Causal attention of q [heads][nq][d] F32 over keys 0..len-1 of the head-first F16 caches k, v [kvHeads][cap][d]:
        /// query t sees keys [floor, len - nq + t]. out [nq][heads][d] F32. With <paramref name="splits"/> &gt; 1 the splits
        /// write <paramref name="partial"/> (<see cref="PartialFloats"/>) and a combine pass writes out. With
        /// <paramref name="dyn"/> (a recorded decode step's parameter block) the length is read on the device. With
        /// <paramref name="cells"/> (one decode row: a sparse attention's selection) the keys are the cache rows listed there,
        /// <paramref name="cellCount"/>[0] of them on the device, each with its value in the mask row
        /// <paramref name="cellMask"/> added (len and floor unused; none added without it: a selection whose kept cells all
        /// hold 0). With <paramref name="bucket"/> &gt; 0 and <paramref name="dyn"/> the launch holds <paramref name="splits"/>
        /// splits (<see cref="Splits"/> of the whole <paramref name="windowKeys"/>-key window) and runs the ones
        /// <see cref="Splits"/> gives the window of <see cref="BucketKeys"/> around the length read on the device plus
        /// <paramref name="lenBias"/> (the keys a verify window's last row has past this row): one recording of a window padded
        /// to the slot's capacity then splits every length as the bucket window around it does. <paramref name="prompt"/>: the
        /// rows are a prompt chunk's (more than any verify window's): its row tiles run heaviest first and its warps skip the key
        /// tiles past their rows (each row's arithmetic unchanged). <paramref name="q8"/>: k and v are q8_0 caches
        /// (<see cref="Q8RowBytes"/> a row), whose values (RN_F16(code * d)) the kernels attend exactly as the F16 kernels attend an
        /// F16 cache holding them.
        /// </summary>
        public unsafe void LaunchAttention(IntPtr q, IntPtr k, IntPtr v, int cap, int nq, int len, int floor, int heads,
            int kvHeads, int headDim, float scale, int splits, IntPtr partial, IntPtr output, IntPtr stream, IntPtr dyn = default,
            IntPtr cells = default, IntPtr cellCount = default, IntPtr cellMask = default, int windowKeys = 0, int bucket = 0, int lenBias = 0,
            bool prompt = false, bool q8 = false)
        {
            int rows = nq * (heads / kvHeads);
            IntPtr a0 = q, a1 = k, a2 = v, a11 = partial, a12 = output, a13 = dyn, a14 = cells, a15 = cellCount, a16 = cellMask;
            int a3 = cap, a4 = nq, a5 = len, a6 = floor, a7 = heads, a8 = kvHeads, a10 = splits, a17 = lenBias, a18 = windowKeys, a19 = bucket,
                a20 = MultiprocessorCount;
            float a9 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16,
                &a17, &a18, &a19, &a20 };
            IntPtr fn = q8
                ? (!prompt ? (headDim == 128 ? _attnQ8128 : _attnQ8256) : headDim == 128 ? _promptQ8128 : _promptQ8256)
                : (!prompt ? (headDim == 128 ? _attn128 : _attn256) : headDim == 128 ? _prompt128 : _prompt256);
            int smem = q8 ? (headDim == 128 ? _smemQ8128 : _smemQ8256) : headDim == 128 ? _smem128 : _smem256;
            uint tiles = (uint)((rows + BlockRows - 1) / BlockRows);
            CudaDriverApi.cuLaunchKernel(fn, tiles * (uint)splits, (uint)kvHeads, 1, 256, 1, 1, (uint)smem, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
            if (splits > 1) LaunchCombine(partial, splits, headDim, output, nq * heads, dyn, lenBias, windowKeys, bucket, rows, kvHeads, stream);
        }

        /// <summary>The grid of <see cref="LaunchPromptAttention"/> for <paramref name="rows"/> query rows (query heads x positions)
        /// of each of <paramref name="kvHeads"/> KV heads whose rows see at least <paramref name="keys"/> keys: its
        /// <see cref="BlockRows"/>-row tiles (all KV heads') fill whole waves of the device a block each, and the last, part-filled
        /// wave's tiles split over their keys into <c>TailSplits</c> blocks each when that ends the launch sooner (each split at
        /// least four key tiles, at most three waves of split blocks). A split count s turns the tail's t tiles into x = t s / SMs
        /// waves of 1/s of a block's time, costed (floor(x) + p(frac(x))) / s plus 2% of a block's time for each extra split (its
        /// Q load, partial write and combine), where a wave filled to f costs p(f) = max(0.8, sqrt(f)) of a full one: its blocks
        /// start as earlier ones end, and the device clocks up to a quarter higher under its power cap while SMs idle (an RTX PRO
        /// 4500 sweep, 300 to 2,048 rows over 30K to 100K keys, interleaved). A launch of fewer row tiles than the device has
        /// multiprocessors (a short chunk, few KV heads) splits them until their blocks fill one wave, up to
        /// <see cref="MaxPromptTailSplits"/>: there each block's time falls with its share of the keys, which the wave cost's floor
        /// and per-split cost do not see. <c>TailSplits</c> 1: no split.</summary>
        public (int FullTiles, int TailSplits) PromptPlan(int rows, int kvHeads, int keys) => PromptPlan(rows, kvHeads, keys, MultiprocessorCount);

        internal static (int FullTiles, int TailSplits) PromptPlan(int rows, int kvHeads, int keys, int sms)
        {
            int total = (rows + BlockRows - 1) / BlockRows * kvHeads, tail = total % sms;
            if (tail == 0) return (total, 1);
            static double Cost(int tail, int s, int sms)
            {
                double x = (double)tail * s / sms, whole = Math.Floor(x), f = x - whole;
                return (whole + (f > 0 ? Math.Max(0.8, Math.Sqrt(f)) : 0)) / s + PromptSplitCost * (s - 1);
            }
            int best = 1;
            double bestCost = Cost(tail, 1, sms);
            for (int s = 2; s <= CostedPromptTailSplits && keys / s >= 4 * TileKeys && tail * s <= 3 * sms; s++)
            {
                double cost = Cost(tail, s, sms);
                if (cost < bestCost - 1e-9) (best, bestCost) = (s, cost);
            }
            // Fewer row tiles than multiprocessors: as many splits as fill one wave. The costed plan had stopped at 6 there:
            // Qwen3.8-27B's 24/4 heads, 17 rows over 100K keys (4 tiles), took 1.58x the 16-bit-run launch's time, 20 splits
            // 0.81x; 8/2 heads of 256 (2 tiles) 3.28x, 41 splits 0.82x (RTX PRO 4500). With whole waves before the tail, filling
            // its wave gained nothing (0.98-1.01x), so those plans stay the costed ones.
            if (total < sms)
            {
                int fill = Math.Min(Math.Min(sms / total, MaxPromptTailSplits), keys / (4 * TileKeys));
                if (fill > best) best = fill;
            }
            return best == 1 ? (total, 1) : (total - tail, best);
        }

        /// <summary>The most splits a tail tile of <see cref="PromptPlan"/> takes (a launch of fewer row tiles than multiprocessors
        /// splits until it fills a wave); the cost model weighs up to <see cref="CostedPromptTailSplits"/>, with the cost of each
        /// split's own work in block times (its Q load, partial write and combine).</summary>
        public const int MaxPromptTailSplits = 64;
        private const int CostedPromptTailSplits = 16;
        private const double PromptSplitCost = 0.02;

        /// <summary>Whether <paramref name="plan"/> is a grid <see cref="LaunchPromptAttention"/> can run for <paramref name="rows"/>
        /// rows of each of <paramref name="kvHeads"/> KV heads: every row tile a block (<c>TailSplits</c> 1, <c>FullTiles</c> all of
        /// them), or some tail tiles split at least twice. A plan of one split with tail tiles would have those blocks write partials
        /// no combine merges (their rows never written, through a null partial pointer), and one of several splits with no tail tile
        /// a combine of no blocks.</summary>
        internal static bool IsPromptPlanOf(int rows, int kvHeads, (int FullTiles, int TailSplits) plan)
        {
            long total = (long)(rows + BlockRows - 1) / BlockRows * kvHeads;
            return plan.FullTiles >= 0 && plan.TailSplits >= 1
                && (plan.TailSplits == 1 ? plan.FullTiles == total : plan.FullTiles < total);
        }

        /// <summary>Floats of the partials a <see cref="PromptPlan"/> grid's tail tiles write: <see cref="BlockRows"/> rows of
        /// headDim + 2 (acc, max, sum) per tail tile and split; 0 without a split.</summary>
        public static long PromptPartialFloats(int rows, int kvHeads, int headDim, (int FullTiles, int TailSplits) plan)
        {
            long total = (long)(rows + BlockRows - 1) / BlockRows * kvHeads;
            return plan.TailSplits <= 1 ? 0 : (total - plan.FullTiles) * BlockRows * plan.TailSplits * (headDim + 2);
        }

        /// <summary>
        /// A prompt chunk's rows on the prompt kernels with F32 accumulation (ts_flash_prompt2_*): the causal attention of
        /// q [heads][nq][d] F32 over keys 0..len-1 of the head-first F16 (or, <paramref name="q8"/>, q8_0) caches k, v
        /// [kvHeads][cap][d]: query t sees keys [floor, len - nq + t]; out [nq][heads][d] F32. <paramref name="plan"/> is
        /// <see cref="PromptPlan"/>(nq * heads / kvHeads, kvHeads, keys), and with a tail split the tail tiles write
        /// <paramref name="partial"/> (<see cref="PromptPartialFloats"/>) that a combine pass merges into out. Every product
        /// accumulates in F32 inside the tensor cores, each 32-key tile's P V flushed into F32 accumulators in one rounding, the
        /// softmax in F32 over F16-rounded probabilities: the same inputs as <see cref="LaunchAttention"/>'s prompt kernels, a
        /// smaller error against an exact reference, other bits (the module's header and section).
        /// </summary>
        public unsafe void LaunchPromptAttention(IntPtr q, IntPtr k, IntPtr v, int cap, int nq, int len, int floor, int heads,
            int kvHeads, int headDim, float scale, (int FullTiles, int TailSplits) plan, IntPtr partial, IntPtr output, IntPtr stream,
            bool q8 = false)
        {
            int rows = nq * (heads / kvHeads), total = (rows + BlockRows - 1) / BlockRows * kvHeads;
            if (!IsPromptPlanOf(rows, kvHeads, plan) || (plan.TailSplits > 1 && partial == IntPtr.Zero))
                throw new ArgumentOutOfRangeException(nameof(plan), $"a plan of {plan.FullTiles} full tiles and {plan.TailSplits} splits for {total} tiles");
            IntPtr a0 = q, a1 = k, a2 = v, a12 = partial, a13 = output;
            int a3 = cap, a4 = nq, a5 = len, a6 = floor, a7 = heads, a8 = kvHeads, a10 = plan.FullTiles, a11 = plan.TailSplits;
            float a9 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13 };
            var (fn, smem) = q8 ? (headDim == 128 ? (_prompt2Q8128, _smemQ8128) : (_prompt2Q8256, _smemQ8256))
                : headDim == 128 ? (_prompt2_128, _smem128) : (_prompt2_256, _smem256);
            uint blocks = (uint)(plan.FullTiles + (total - plan.FullTiles) * plan.TailSplits);
            CudaDriverApi.cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, (uint)smem, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
            if (plan.TailSplits <= 1) return;
            IntPtr c0 = partial, c7 = output;
            int c1 = nq, c2 = heads, c3 = kvHeads, c4 = plan.FullTiles, c5 = plan.TailSplits, c6 = headDim;
            void** cargs = stackalloc void*[] { &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7 };
            CudaDriverApi.cuLaunchKernel(_prompt2Combine, (uint)((total - plan.FullTiles) * BlockRows), 1, 1, (uint)headDim, 1, 1, 0, stream,
                (IntPtr)cargs, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>
        /// The 8-bit prompt attention of one sequence's prompt chunk (<see cref="Runs8BitPrompt"/>): the band of keys
        /// [floor &amp; ~31, len) of the head-first F16 caches k, v [kvHeads][cap][256] into <paramref name="band"/>
        /// (<see cref="Attn8BandBytes"/>(<paramref name="bandKeys"/>, kvHeads) bytes, <paramref name="bandKeys"/> at least
        /// len and a multiple of 32), then the causal attention of q [heads][nq][256] F32 over it: query t sees keys
        /// [floor, len - nq + t]; out [nq][heads][256] F32. With <paramref name="splits"/> &gt; 1 the splits write
        /// <paramref name="partial"/> (<see cref="PartialFloats"/>) and the combine pass writes out. The rows' operands and
        /// splits are <see cref="LaunchAttention"/>'s (prompt); the arithmetic is 8-bit (the module header).
        /// </summary>
        public unsafe void LaunchPromptAttention8(IntPtr q, IntPtr k, IntPtr v, int cap, int nq, int len, int floor, int heads,
            int kvHeads, float scale, int splits, IntPtr band, int bandKeys, IntPtr partial, IntPtr output, IntPtr stream)
        {
            LaunchPromptBand(k, v, cap, len, floor, kvHeads, band, bandKeys, stream);
            var layout = Attn8BandLayout(bandKeys, kvHeads);
            IntPtr a0 = q, a1 = band + (nint)layout.K8, a2 = band + (nint)layout.KScales, a3 = band + (nint)layout.VT,
                a4 = band + (nint)layout.VScales, a13 = partial, a14 = output;
            int a5 = bandKeys, a6 = nq, a7 = len, a8 = floor, a9 = heads, a10 = kvHeads, a12 = splits;
            float a11 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14 };
            int rows = nq * (heads / kvHeads);
            uint tiles = (uint)((rows + BlockRows - 1) / BlockRows);
            CudaDriverApi.cuLaunchKernel(_attn8_256, tiles * (uint)splits, (uint)kvHeads, 1, 256, 1, 1, Attn8SharedBytes, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
            if (splits > 1) LaunchCombine(partial, splits, Attn8HeadDim, output, nq * heads, IntPtr.Zero, 0, 0, 0, rows, kvHeads, stream);
        }

        /// <summary>The band alone (<see cref="LaunchPromptAttention8"/>'s first launch; tests read it back).</summary>
        internal unsafe void LaunchPromptBand(IntPtr k, IntPtr v, int cap, int len, int floor, int kvHeads, IntPtr band, int bandKeys,
            IntPtr stream)
        {
            if (bandKeys < len || bandKeys % TileKeys != 0 || floor < 0 || floor >= len)
                throw new ArgumentOutOfRangeException(nameof(bandKeys), $"a band of {bandKeys} keys for keys [{floor}, {len})");
            var layout = Attn8BandLayout(bandKeys, kvHeads);
            IntPtr b0 = k, b1 = v, b6 = band + (nint)layout.K8, b7 = band + (nint)layout.KScales, b8 = band + (nint)layout.VT,
                b9 = band + (nint)layout.VScales;
            int b2 = cap, b3 = len, b4 = floor, b5 = bandKeys;
            void** args = stackalloc void*[] { &b0, &b1, &b2, &b3, &b4, &b5, &b6, &b7, &b8, &b9 };
            uint tiles = (uint)((len + TileKeys - 1) / TileKeys - floor / TileKeys);
            CudaDriverApi.cuLaunchKernel(_band256, tiles, (uint)kvHeads, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>The combine over the splits the attention launch ran (its arguments: <paramref name="ruleRows"/> the rows its
        /// splits were counted for).</summary>
        private unsafe void LaunchCombine(IntPtr partial, int splits, int headDim, IntPtr output, int outputs, IntPtr dyn, int lenBias,
            int windowKeys, int bucket, int ruleRows, int kvHeads, IntPtr stream)
        {
            IntPtr c0 = partial, c3 = output, c4 = dyn;
            int c1 = splits, c2 = headDim, c5 = lenBias, c6 = windowKeys, c7 = bucket, c8 = ruleRows, c9 = kvHeads, c10 = MultiprocessorCount;
            void** cargs = stackalloc void*[] { &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9, &c10 };
            CudaDriverApi.cuLaunchKernel(_combine, (uint)outputs, 1, 1, (uint)headDim, 1, 1, 0, stream, (IntPtr)cargs,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>
        /// A verify window's <paramref name="rows"/> rows, each as the decode row at its position, in one launch: row z of q
        /// [rows][heads][d] F32 attends keys [floor, len - (rows - 1 - z)) of the head-first F16 caches (len read from
        /// <paramref name="dyn"/>[0] on the device when given), into out [rows][heads][d] F32. Each row's arithmetic is a
        /// one-row <see cref="LaunchAttention"/>'s; the partials hold <see cref="PartialFloats"/>(rows, ...). With
        /// <paramref name="cells"/> (a sparse attention's selection) row z's keys are cells[z * cellStride ..], cellCount[z] of
        /// them, its mask row cellMask[z * cellStride ..] (or none); with <paramref name="bucket"/> &gt; 0 row z runs the splits
        /// <see cref="Splits"/> gives <see cref="BucketKeys"/> of <paramref name="windowKeys"/> and its count (the decode row's over
        /// the same cells), never more than <paramref name="splits"/>, and bucket 0 runs every row on them. Without a selection
        /// the rows share the launch's row tiles (ts_flash_decode_window_f16_d*: every key tile read once for the window's rows,
        /// each row still its one-row launch's bits); with one, a row of the grid each. Without a selection and with
        /// <paramref name="bucket"/> &gt; 0, row z runs the splits <see cref="Splits"/> gives <see cref="BucketKeys"/> of
        /// <paramref name="windowKeys"/> and its own length (the decode row's at its position, recorded or not), never more than
        /// <paramref name="splits"/> (the last row's); bucket 0 runs every row on <paramref name="splits"/>. <paramref name="q8"/>: k and
        /// v are q8_0 caches (as <see cref="LaunchAttention"/>).
        /// </summary>
        public unsafe void LaunchDecodeRows(IntPtr q, IntPtr k, IntPtr v, int cap, int rows, int len, int floor, int heads,
            int kvHeads, int headDim, float scale, int splits, IntPtr partial, IntPtr output, IntPtr stream, IntPtr dyn = default,
            IntPtr cells = default, IntPtr cellCount = default, IntPtr cellMask = default, int cellStride = 0, int windowKeys = 0, int bucket = 0,
            bool q8 = false)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a11 = partial, a12 = output, a13 = dyn, a14 = cells, a15 = cellCount, a16 = cellMask;
            int a3 = cap, a4 = rows, a5 = len, a6 = floor, a7 = heads, a8 = kvHeads, a10 = splits, a17 = cellStride, a18 = windowKeys, a19 = bucket,
                a20 = MultiprocessorCount;
            float a9 = scale;
            if (cells == IntPtr.Zero && !RowGridForTests)
            {
                void** wargs = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a18, &a19, &a20 };
                var (wfn, wsmem) = q8 ? (headDim == 128 ? (_windowQ8128, _smemQ8128) : (_windowQ8256, _smemQ8256))
                    : headDim == 128 ? (_window128, _smem128) : (_window256, _smem256);
                uint windowTiles = (uint)((rows * (heads / kvHeads) + BlockRows - 1) / BlockRows);
                CudaDriverApi.cuLaunchKernel(wfn, windowTiles * (uint)splits, (uint)kvHeads, 1, 256, 1, 1, (uint)wsmem, stream, (IntPtr)wargs,
                    IntPtr.Zero).ThrowOnError();
            }
            else
            {
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17,
                    &a18, &a19, &a20 };
                var (fn, smem) = q8 ? (headDim == 128 ? (_rowsQ8128, _smemQ8128) : (_rowsQ8256, _smemQ8256))
                    : headDim == 128 ? (_rows128, _smem128) : (_rows256, _smem256);
                uint tiles = (uint)((heads / kvHeads + BlockRows - 1) / BlockRows);
                CudaDriverApi.cuLaunchKernel(fn, tiles * (uint)splits, (uint)kvHeads, (uint)rows, 256, 1, 1, (uint)smem, stream, (IntPtr)args,
                    IntPtr.Zero).ThrowOnError();
            }
            if (splits <= 1) return;
            if (cells != IntPtr.Zero && bucket > 0)
            {
                // Each selection's row combines the splits its own listed count gave it.
                IntPtr l0 = partial, l3 = output, l4 = cellCount;
                int l1 = splits, l2 = headDim, l5 = heads, l6 = windowKeys, l7 = bucket, l8 = heads / kvHeads, l9 = kvHeads, l10 = MultiprocessorCount;
                void** largs = stackalloc void*[] { &l0, &l1, &l2, &l3, &l4, &l5, &l6, &l7, &l8, &l9, &l10 };
                CudaDriverApi.cuLaunchKernel(_listedCombine, (uint)(rows * heads), 1, 1, (uint)headDim, 1, 1, 0, stream, (IntPtr)largs,
                    IntPtr.Zero).ThrowOnError();
                return;
            }
            if (cells != IntPtr.Zero)
            {
                LaunchCombine(partial, splits, headDim, output, rows * heads, dyn, 0, windowKeys, bucket, heads / kvHeads, kvHeads, stream);
                return;
            }
            // Each row combines the splits its own length gave it.
            IntPtr c0 = partial, c3 = output, c4 = dyn;
            int c1 = splits, c2 = headDim, c5 = len, c6 = rows, c7 = heads, c8 = windowKeys, c9 = bucket, c10 = heads / kvHeads, c11 = kvHeads,
                c12 = MultiprocessorCount;
            void** cargs = stackalloc void*[] { &c0, &c1, &c2, &c3, &c4, &c5, &c6, &c7, &c8, &c9, &c10, &c11, &c12 };
            CudaDriverApi.cuLaunchKernel(_windowCombine, (uint)(rows * heads), 1, 1, (uint)headDim, 1, 1, 0, stream, (IntPtr)cargs,
                IntPtr.Zero).ThrowOnError();
        }

        /// <summary>Tests and A/B: <see cref="LaunchDecodeRows"/> runs a window's rows on the row grid (a row a z-slice) without a
        /// selection too.</summary>
        internal bool RowGridForTests { get; set; }

        /// <summary>The module this process loads (the build output's PTX, or the fatbin beside it),
        /// whether or not it is current; see <see cref="CudaPrefillModule.IsCurrent"/>. Requires a
        /// current context.</summary>
        public static CudaFlashPrefillKernels Create() => FromModule(LoadModule(PtxFileName));

        /// <summary>Kernels over an already loaded module (tests that load a specific image), which
        /// they own. Requires the module's context to be current.</summary>
        internal static CudaFlashPrefillKernels FromModule(CudaModule module) => Own(module, m => new CudaFlashPrefillKernels(m));

        /// <summary>The current module, or false with the reason it cannot be used.</summary>
        public static bool TryCreate(out CudaFlashPrefillKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaFlashPrefillKernels(m), out kernels, out reason);
    }
}
