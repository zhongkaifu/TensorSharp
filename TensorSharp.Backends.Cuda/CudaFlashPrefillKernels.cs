// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the tensor-core flash prefill attention kernels
// (native/kernels/tensorsharp_flash_prefill.cu, compiled to tensorsharp_flash_prefill.ptx): the
// attention CudaGraphExecutor runs a prompt chunk's rows on, over a head-first F16 cache.
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
        public const int CurrentVersion = 11;

        /// <summary>Query rows (query heads x positions of one KV head) one block takes, and keys per tile.</summary>
        public const int BlockRows = 128, TileKeys = 32;

        private const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
        private readonly IntPtr _attn128, _attn256, _rows128, _rows256, _window128, _window256, _combine, _windowCombine;
        private readonly IntPtr _prompt128, _prompt256;
        private readonly IntPtr _band256, _attn8_256;
        private readonly int _smem128, _smem256;

        /// <summary>The 8-bit prompt attention's head width, and the fewest keys a chunk attends before it takes it (the direct
        /// engine's Attn8MinKeys: below them the band is most of the work).</summary>
        public const int Attn8HeadDim = 256, Attn8MinKeys = 1024;

        /// <summary>The fewest prompt rows the 8-bit attention takes: below them the band (converted once per layer whatever
        /// the rows) costs more than the rows save. Provisional until measured per architecture (an RTX PRO 4500 row sweep).</summary>
        public const int Attn8MinRows = 256;

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
            CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device).ThrowOnError();
            MultiprocessorCount = sms;
            // The 8-bit prompt attention: only an sm_120a / sm_121a image carries its block-scaled E4M3 MMA.
            Attn8Code = ReadAttn8Caps(module);
            if (Attn8Code)
            {
                _band256 = module.GetFunction("ts_flash_prompt_band_f16_d256");
                _attn8_256 = module.GetFunction("ts_flash_prompt_i8e4_d256");
            }
        }

        public int MultiprocessorCount { get; }

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
        /// V tiles, <see cref="TileKeys"/> rows of headDim + 8 halves each.</summary>
        public static int SharedBytes(int headDim) => 2 * 2 * TileKeys * (headDim + 8) * 2;

        /// <summary>Whether <see cref="LaunchAttention"/> runs here for <paramref name="headDim"/>-wide heads (sm_80 code).</summary>
        public bool RunsAttention(int headDim) => IsCurrent && ArchGuardedKernelsUsable && (headDim == 128 || headDim == 256);

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
        /// tiles past their rows (each row's arithmetic unchanged).
        /// </summary>
        public unsafe void LaunchAttention(IntPtr q, IntPtr k, IntPtr v, int cap, int nq, int len, int floor, int heads,
            int kvHeads, int headDim, float scale, int splits, IntPtr partial, IntPtr output, IntPtr stream, IntPtr dyn = default,
            IntPtr cells = default, IntPtr cellCount = default, IntPtr cellMask = default, int windowKeys = 0, int bucket = 0, int lenBias = 0,
            bool prompt = false)
        {
            int rows = nq * (heads / kvHeads);
            IntPtr a0 = q, a1 = k, a2 = v, a11 = partial, a12 = output, a13 = dyn, a14 = cells, a15 = cellCount, a16 = cellMask;
            int a3 = cap, a4 = nq, a5 = len, a6 = floor, a7 = heads, a8 = kvHeads, a10 = splits, a17 = lenBias, a18 = windowKeys, a19 = bucket,
                a20 = MultiprocessorCount;
            float a9 = scale;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16,
                &a17, &a18, &a19, &a20 };
            IntPtr fn = !prompt ? (headDim == 128 ? _attn128 : _attn256) : headDim == 128 ? _prompt128 : _prompt256;
            int smem = headDim == 128 ? _smem128 : _smem256;
            uint tiles = (uint)((rows + BlockRows - 1) / BlockRows);
            CudaDriverApi.cuLaunchKernel(fn, tiles * (uint)splits, (uint)kvHeads, 1, 256, 1, 1, (uint)smem, stream, (IntPtr)args,
                IntPtr.Zero).ThrowOnError();
            if (splits > 1) LaunchCombine(partial, splits, headDim, output, nq * heads, dyn, lenBias, windowKeys, bucket, rows, kvHeads, stream);
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
        /// them, its mask row cellMask[z * cellStride ..] (or none), every row on <paramref name="splits"/>. Without a selection
        /// the rows share the launch's row tiles (ts_flash_decode_window_f16_d*: every key tile read once for the window's rows,
        /// each row still its one-row launch's bits); with one, a row of the grid each. Without a selection and with
        /// <paramref name="bucket"/> &gt; 0, row z runs the splits <see cref="Splits"/> gives <see cref="BucketKeys"/> of
        /// <paramref name="windowKeys"/> and its own length (the decode row's at its position, recorded or not), never more than
        /// <paramref name="splits"/> (the last row's); bucket 0 runs every row on <paramref name="splits"/>.
        /// </summary>
        public unsafe void LaunchDecodeRows(IntPtr q, IntPtr k, IntPtr v, int cap, int rows, int len, int floor, int heads,
            int kvHeads, int headDim, float scale, int splits, IntPtr partial, IntPtr output, IntPtr stream, IntPtr dyn = default,
            IntPtr cells = default, IntPtr cellCount = default, IntPtr cellMask = default, int cellStride = 0, int windowKeys = 0, int bucket = 0)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a11 = partial, a12 = output, a13 = dyn, a14 = cells, a15 = cellCount, a16 = cellMask;
            int a3 = cap, a4 = rows, a5 = len, a6 = floor, a7 = heads, a8 = kvHeads, a10 = splits, a17 = cellStride, a18 = windowKeys, a19 = bucket,
                a20 = MultiprocessorCount;
            float a9 = scale;
            if (cells == IntPtr.Zero && !RowGridForTests)
            {
                void** wargs = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a18, &a19, &a20 };
                var (wfn, wsmem) = headDim == 128 ? (_window128, _smem128) : (_window256, _smem256);
                uint windowTiles = (uint)((rows * (heads / kvHeads) + BlockRows - 1) / BlockRows);
                CudaDriverApi.cuLaunchKernel(wfn, windowTiles * (uint)splits, (uint)kvHeads, 1, 256, 1, 1, (uint)wsmem, stream, (IntPtr)wargs,
                    IntPtr.Zero).ThrowOnError();
            }
            else
            {
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a12, &a13, &a14, &a15, &a16, &a17,
                    &a18, &a19, &a20 };
                var (fn, smem) = headDim == 128 ? (_rows128, _smem128) : (_rows256, _smem256);
                uint tiles = (uint)((heads / kvHeads + BlockRows - 1) / BlockRows);
                CudaDriverApi.cuLaunchKernel(fn, tiles * (uint)splits, (uint)kvHeads, (uint)rows, 256, 1, 1, (uint)smem, stream, (IntPtr)args,
                    IntPtr.Zero).ThrowOnError();
            }
            if (splits <= 1) return;
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
