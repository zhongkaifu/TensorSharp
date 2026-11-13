// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// NVFP4 products of the direct-CUDA backend's quantized matmul on GPUs whose image carries the block-scaled FP4
    /// tensor cores (tensorsharp_fp4_kernels: an sm_120a / sm_121a build). A resident NVFP4 matrix is kept as two planes,
    /// codes [rows][k/2] then UE4M3 scales [rows][k/16] (<see cref="Repack"/>): up to 8 rows run the q8 matvec, more the
    /// FP4 GEMMs (split over K for few rows, TMA-fed for prompt rows).
    /// </summary>
    internal sealed unsafe class CudaFp4Kernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_fp4_kernels.ptx";
        public const string CapsKernelName = "ts_fp4_caps";

        /// <summary>TS_FP4_VERSION in tensorsharp_fp4_kernels.cu: bump both together whenever a kernel is added or
        /// changes its signature.</summary>
        public const int CurrentVersion = 3;

        /// <summary>K values one GEMM stage takes: a planar matrix's row width must be a multiple of it.</summary>
        public const int StageK = 256;

        /// <summary>Weight rows one GEMM tile takes.</summary>
        public const int TileRows = 128;

        /// <summary>Activation rows the q8 matvec takes; more run on the FP4 GEMMs.</summary>
        public const int GemvMaxRows = 8;

        /// <summary>Fewest activation rows the TMA-fed GEMM takes.</summary>
        public const int TmaMinRows = 65;

        /// <summary>Widest activation row the NVFP4 quantizer stages in shared memory.</summary>
        public const int MaxQuantRowFloats = 20 * 1024;

        private const int Block = 256;
        private const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

        private readonly IntPtr _repack, _quantQ81, _quantNvfp4, _quantSwiGlu, _gemm32, _gemm64, _gemm128, _tma256, _tma128, _getRows;
        private readonly IntPtr[] _gemv = new IntPtr[GemvMaxRows + 1];
        private readonly int[] _gemvRows = new int[GemvMaxRows + 1];
        private readonly int _smem32, _smem64, _smem128, _smemTma256, _smemTma128;

        private CudaFp4Kernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
            if (!IsCurrent) return;
            int* host = stackalloc int[16];
            RunProbe(module.GetFunction("ts_fp4_mma_caps"), host, 1);
            HasTensorCores = host[0] == 1;
            if (!HasTensorCores) return;
            CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
            CudaDriverApi.cuDeviceGetAttribute(out int sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, device).ThrowOnError();
            MultiprocessorCount = sms;
            _repack = module.GetFunction("ts_fp4_repack_nvfp4");
            _quantQ81 = module.GetFunction("ts_fp4_quantize_q81_f32");
            _quantNvfp4 = module.GetFunction("ts_fp4_quantize_nvfp4_f32");
            _quantSwiGlu = module.GetFunction("ts_fp4_quantize_swiglu_nvfp4_f32");
            _gemm32 = module.GetFunction("ts_fp4_gemm_w4a4_32");
            _gemm64 = module.GetFunction("ts_fp4_gemm_w4a4_64");
            _gemm128 = module.GetFunction("ts_fp4_gemm_w4a4_128");
            _tma256 = module.GetFunction("ts_fp4_gemm_w4a4_tma256");
            _tma128 = module.GetFunction("ts_fp4_gemm_w4a4_tma128");
            _getRows = module.GetFunction("ts_fp4_get_rows_f32");
            for (int n = 1; n <= GemvMaxRows; n++)
                _gemv[n] = module.GetFunction($"ts_fp4_gemv_nvfp4_{n}");
            RunProbe(module.GetFunction("ts_fp4_gemv_rows_per_warp"), host, GemvMaxRows + 1);
            for (int n = 1; n <= GemvMaxRows; n++)
                _gemvRows[n] = host[n];
            RunProbe(module.GetFunction("ts_fp4_gemm_smem_bytes"), host, 3);
            (_smem32, _smem64, _smem128) = (host[0], host[1], host[2]);
            RunProbe(module.GetFunction("ts_fp4_gemm_tma_smem_bytes"), host, 2);
            (_smemTma256, _smemTma128) = (host[0], host[1]);
            CudaDriverApi.cuFuncSetAttribute(_quantNvfp4, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, MaxQuantRowFloats * 4).ThrowOnError();
            CudaDriverApi.cuFuncSetAttribute(_quantSwiGlu, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, MaxQuantRowFloats * 4).ThrowOnError();
            foreach (var (fn, smem) in new[] { (_gemm32, _smem32), (_gemm64, _smem64), (_gemm128, _smem128), (_tma256, _smemTma256), (_tma128, _smemTma128) })
                CudaDriverApi.cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem).ThrowOnError();
        }

        /// <summary>Whether the loaded image carries the FP4 tensor-core bodies (an sm_120a / sm_121a image).</summary>
        public bool HasTensorCores { get; }

        public int MultiprocessorCount { get; }

        /// <summary>The module for the current context, or false with the reason it cannot run the products.</summary>
        public static bool TryCreate(out CudaFp4Kernels kernels, out string reason)
        {
            if (!TryCreate(PtxFileName, m => new CudaFp4Kernels(m), out kernels, out reason))
                return false;
            if (kernels.HasTensorCores)
                return true;
            reason = "the loaded tensorsharp_fp4_kernels image has no FP4 tensor-core code for this GPU (an sm_120a / sm_121a build carries it)";
            kernels.Dispose();
            kernels = null;
            return false;
        }

        /// <summary>Whether an NVFP4 matrix of <paramref name="k"/>-wide rows is kept as planes (and runs here).</summary>
        public static bool TakesWidth(long k) => k > 0 && k % StageK == 0 && k <= MaxQuantRowFloats;

        private static void RunProbe(IntPtr fn, int* host, int count)
        {
            CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr((ulong)count * 4)).ThrowOnError();
            try
            {
                IntPtr p = probe;
                void** args = stackalloc void*[] { &p };
                CudaDriverApi.cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr((ulong)count * 4)).ThrowOnError();
            }
            finally
            {
                CudaDriverApi.cuMemFree(probe);
            }
        }

        private static void Launch(IntPtr fn, uint gx, uint gy, uint gz, int block, int smem, IntPtr stream, void** args)
            => CudaDriverApi.cuLaunchKernel(fn, gx, gy, gz, (uint)block, 1, 1, (uint)smem, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();

        private static uint CeilDiv(long n, long d) => (uint)Math.Max(1, (n + d - 1) / d);

        /// <summary>The scale plane of a planar matrix of <paramref name="rows"/> x <paramref name="k"/>.</summary>
        public static IntPtr ScalePlane(IntPtr planes, long rows, long k) => planes + (nint)(rows * k / 2);

        /// <summary>ggml block_nvfp4 rows [rows][k / 64 x 36 bytes] at <paramref name="blocks"/> to the planes at
        /// <paramref name="planes"/> (codes, then scales; the same bytes).</summary>
        public void Repack(IntPtr blocks, IntPtr planes, int rows, int k, IntPtr stream)
        {
            IntPtr a0 = blocks, a2 = planes, a3 = ScalePlane(planes, rows, k);
            long a1 = (long)(k / 64) * 36;
            int a4 = rows, a5 = k, a6 = 0, a7 = 1, a8 = 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
            Launch(_repack, CeilDiv((long)rows * (k / 64), Block), 1, 1, Block, 0, stream, args);
        }

        /// <summary>Scratch bytes of the matvec's q8 activations (int8 values, then one F32 scale per 32).</summary>
        public static long GemvScratchBytes(int rows, int k) => (long)rows * k + (long)rows * (k / 32) * 4;

        /// <summary>y[rows, outDim] = x[rows, k] W^T for 1..<see cref="GemvMaxRows"/> rows: x quantized to q8 (a scale per 32
        /// values) in <paramref name="scratch"/>, then the planar matvec. Each row computes alike whatever the row count.</summary>
        public void Gemv(IntPtr planes, int outDim, int k, IntPtr x, int rows, IntPtr scratch, IntPtr y, IntPtr stream)
        {
            IntPtr xq = scratch, xd = scratch + (nint)((long)rows * k);
            {
                IntPtr a0 = x, a1 = xq, a2 = xd; int a3 = k, a4 = rows;
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
                Launch(_quantQ81, CeilDiv((long)rows * (k / 32) * 32, Block), 1, 1, Block, 0, stream, args);
            }
            {
                IntPtr a0 = planes, a1 = ScalePlane(planes, outDim, k), a4 = xq, a5 = xd, a7 = y;
                int a2 = outDim, a3 = k; float a6 = 1.0f;
                void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
                Launch(_gemv[rows], CeilDiv(outDim, 4 * _gemvRows[rows]), 1, 1, 128, 0, stream, args);
            }
        }

        /// <summary>Activation code bytes, scale bytes and row-scale bytes of <paramref name="rows"/> NVFP4 rows.</summary>
        public static (long Codes, long Scales, long RowScales) ActivationBytes(long rows, long k)
            => (rows * k / 2, rows * k / 16, rows * 4);

        /// <summary>F32 rows [rows][k] to NVFP4 planes (codes [rows][k/2], scales [rows][k/16]) and per-row scales.</summary>
        public void Quantize(IntPtr x, IntPtr codes, IntPtr scales, IntPtr rowScale, int k, int rows, IntPtr stream)
        {
            IntPtr a0 = x, a1 = codes, a2 = scales, a3 = rowScale; int a4 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(_quantNvfp4, (uint)rows, 1, 1, 512, k * 4, stream, args);
        }

        /// <summary>NVFP4 planes of silu(gate) * up for fused gate|up rows [rows][2k] (gate first): the planes
        /// <see cref="Quantize"/> makes of that activation, which is never stored as F32.</summary>
        public void QuantizeSwiGlu(IntPtr gateUp, IntPtr codes, IntPtr scales, IntPtr rowScale, int k, int rows, IntPtr stream)
        {
            IntPtr a0 = gateUp, a1 = codes, a2 = scales, a3 = rowScale; int a4 = k;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
            Launch(_quantSwiGlu, (uint)rows, 1, 1, 512, k * 4, stream, args);
        }

        /// <summary>Whether <paramref name="m"/> rows of a product run on the TMA-fed GEMM (16-byte aligned planes).</summary>
        public static bool TakesTma(int m, int outDim, int k, IntPtr planes, IntPtr xCodes, IntPtr xScales)
            => m >= TmaMinRows && k % 128 == 0 && outDim % 16 == 0
               && ((planes | ScalePlane(planes, outDim, k) | xCodes | xScales) & 15) == 0;

        /// <summary>
        /// Split-K factor of a cp.async GEMM (1 for the 128-column tiles): K is split as far as the CTAs still fit one wave
        /// (a second, partly filled wave costs more than the extra splits save), at most <paramref name="maxSplits"/>.
        /// </summary>
        public int Splits(int outDim, int k, int m, int maxSplits)
        {
            int bn = TileColumns(m);
            if (bn == 128) return 1;
            int tilesX = (outDim + TileRows - 1) / TileRows, tilesY = (m + bn - 1) / bn;
            int stages = k / StageK, splits = 1;
            while (tilesX * tilesY * splits * 2 <= MultiprocessorCount && splits * 2 <= stages / 2 && splits * 2 <= maxSplits)
                splits *= 2;
            return splits;
        }

        /// <summary>Output tiles of a cp.async GEMM (the split-K counters it needs).</summary>
        public static int Tiles(int outDim, int m)
        {
            int bn = TileColumns(m);
            return ((outDim + TileRows - 1) / TileRows) * ((m + bn - 1) / bn);
        }

        private static int TileColumns(int m) => m <= 32 ? 32 : m <= 64 ? 64 : 128;

        /// <summary>
        /// y[m, outDim] = alpha * rowScale * (x W^T) on the cp.async GEMM. With <paramref name="splits"/> &gt; 1,
        /// <paramref name="partial"/> holds splits * m * outDim floats and <paramref name="counters"/> one zeroed int per
        /// output tile (the last CTA of a tile resets its counter).
        /// </summary>
        public void Gemm(IntPtr planes, int outDim, int k, IntPtr xCodes, IntPtr xScales, IntPtr rowScale, int m, float alpha,
            IntPtr y, IntPtr partial, IntPtr counters, int splits, IntPtr stream)
        {
            int bn = TileColumns(m);
            int tilesX = (outDim + TileRows - 1) / TileRows, tilesY = (m + bn - 1) / bn;
            IntPtr a0 = planes, a1 = ScalePlane(planes, outDim, k), a4 = xCodes, a5 = xScales, a6 = rowScale, a9 = y, a10 = partial, a11 = counters;
            int a2 = outDim, a3 = k, a7 = m; float a8 = alpha;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11 };
            var (fn, smem) = bn == 32 ? (_gemm32, _smem32) : bn == 64 ? (_gemm64, _smem64) : (_gemm128, _smem128);
            // The 128-column kernel walks activation tiles fastest (weight tiles shared in L2).
            if (bn == 128)
                Launch(fn, (uint)tilesY, (uint)tilesX, 1, 256, smem, stream, args);
            else
                Launch(fn, (uint)tilesX, (uint)tilesY, (uint)splits, 256, smem, stream, args);
        }

        /// <summary>y[m, outDim] = alpha * rowScale * (x W^T) on the TMA-fed GEMM (<see cref="TakesTma"/>): the products and
        /// summation order of <see cref="Gemm"/> unsplit. Token tiles of 256 rows from 768 rows, else 128.</summary>
        public void GemmTma(IntPtr planes, int outDim, int k, IntPtr xCodes, IntPtr xScales, IntPtr rowScale, int m, float alpha,
            IntPtr y, IntPtr stream)
        {
            int bm = m >= 768 ? 256 : 128;
            // CUtensorMap is 128 bytes and 64-byte aligned: xCodes, xScales, wCodes, wScales (codes in 64-byte boxes with
            // the 64-byte swizzle, scales in 16-byte boxes).
            byte* raw = stackalloc byte[4 * 128 + 64];
            byte* maps = (byte*)(((nint)raw + 63) & ~(nint)63);
            EncodeTma(maps, xCodes, k / 2, m, 64, (uint)bm, 2);
            EncodeTma(maps + 128, xScales, k / 16, m, 16, (uint)bm, 0);
            EncodeTma(maps + 256, planes, k / 2, outDim, 64, 128, 2);
            EncodeTma(maps + 384, ScalePlane(planes, outDim, k), k / 16, outDim, 16, 128, 0);
            IntPtr a3 = rowScale, a6 = y;
            int a1 = outDim, a2 = k, a4 = m; float a5 = alpha;
            void** args = stackalloc void*[] { maps, &a1, &a2, &a3, &a4, &a5, &a6 };
            Launch(bm == 256 ? _tma256 : _tma128, (uint)((m + bm - 1) / bm), (uint)((outDim + 127) / 128), 1, bm == 256 ? 384 : 288,
                bm == 256 ? _smemTma256 : _smemTma128, stream, args);
        }

        private static void EncodeTma(byte* map, IntPtr address, long columns, long rows, uint boxColumns, uint boxRows, int swizzle)
        {
            ulong* dims = stackalloc ulong[2] { (ulong)columns, (ulong)rows };
            ulong* strides = stackalloc ulong[1] { (ulong)columns };
            uint* box = stackalloc uint[2] { boxColumns, boxRows };
            uint* elementStrides = stackalloc uint[2] { 1, 1 };
            CudaDriverApi.cuTensorMapEncodeTiled(map, 0, 2, address, dims, strides, box, elementStrides, 0, swizzle, 0, 0).ThrowOnError();
        }

        /// <summary>Rows <paramref name="indices"/> (int32, or F32 when <paramref name="floatIndices"/>) of a planar matrix of
        /// <paramref name="rows"/> x <paramref name="k"/> as F32 rows.</summary>
        public void GetRows(IntPtr planes, int rows, int k, IntPtr indices, bool floatIndices, int count, IntPtr output, IntPtr stream)
        {
            IntPtr a0 = planes, a3 = indices, a5 = output;
            int a1 = rows, a2 = k, a4 = floatIndices ? 1 : 0;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
            Launch(_getRows, (uint)count, 1, 1, Block, 0, stream, args);
        }
    }
}
