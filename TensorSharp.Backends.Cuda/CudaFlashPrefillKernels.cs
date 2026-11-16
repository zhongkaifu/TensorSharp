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
        public const int CurrentVersion = 9;

        /// <summary>Query rows (query heads x positions of one KV head) one block takes, and keys per tile.</summary>
        public const int BlockRows = 128, TileKeys = 32;

        private const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;
        private readonly IntPtr _attn128, _attn256, _rows128, _rows256, _window128, _window256, _combine;
        private readonly IntPtr _prompt128, _prompt256;
        private readonly int _smem128, _smem256;

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
        }

        public int MultiprocessorCount { get; }

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
        /// one-row <see cref="LaunchAttention"/>'s with the same <paramref name="splits"/>; the partials hold
        /// <see cref="PartialFloats"/>(rows, ...). With <paramref name="cells"/> (a sparse attention's selection) row z's
        /// keys are cells[z * cellStride ..], cellCount[z] of them, its mask row cellMask[z * cellStride ..] (or none).
        /// Without a selection the rows share the launch's row tiles (ts_flash_decode_window_f16_d*: every key tile read once
        /// for the window's rows, each row still its one-row launch's bits); with one, a row of the grid each.
        /// <paramref name="windowKeys"/> and <paramref name="bucket"/> as <see cref="LaunchAttention"/>'s, for every row the
        /// splits of the window's length (a row's as the decode row at its position).
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
            if (splits > 1) LaunchCombine(partial, splits, headDim, output, rows * heads, dyn, 0, windowKeys, bucket, heads / kvHeads, kvHeads, stream);
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
