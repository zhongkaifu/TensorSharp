// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>NInfer prefill GEMMs on the TMA-fed, warp-specialized FP4 tiles
/// (ts_q35_gemm_w4a4_tma&lt;bm&gt;_ni_*), whose epilogue applies the BF16 storage boundary
/// (and, for gate|up, SwiGLU and the down projection's NVFP4 activation quantization)
/// that NInfer otherwise runs as separate full passes over F32 buffers
/// (<c>ts_ni_round</c>, <c>ts_ni_swiglu</c>, <c>ts_ni_quant_nvfp4</c>). Same arithmetic:
/// the accumulators are those of the cp.async tiles bit for bit, and the epilogue applies
/// the very roundings, SiLU and quantizer the separate kernels apply. RTX 5090, NInfer
/// Qwen3.8-27B TTFT 8K / 32K / 60K: 0.918 / 4.377 / 10.169 s -> 0.742 / 3.692 / 8.880 s,
/// identical answers.</summary>
internal sealed unsafe partial class Q35Kernels
{
    internal enum NinferPrefillEpilogue { Plain = 0, Round = 1, SwiGluRoundedInputs = 2, SwiGlu = 3 }
    private const int SwiGluQuantRoundedInputs = 4, SwiGluQuant = 5;
    private const int NinferTmaEpilogueCount = 6;
    private static readonly string[] NinferTmaEpilogues = { "plain", "round", "swiglu_rin", "swiglu", "swiglu_rin_q", "swiglu_q" };
    // [token tile: 0 = 256, 1 = 128][epilogue]
    private readonly IntPtr[][] _niTma = new IntPtr[2][];
    private readonly int[] _niTmaSmem = new int[2 * NinferTmaEpilogueCount];
    private int _niTmaState; // 0 unresolved, 1 available, -1 absent
    internal long NinferPrefillEpilogueSubmissions { get; private set; }

    /// <summary>Minimum activation rows: below this NInfer prefill keeps the split-K GEMV/GEMM routes.</summary>
    internal const int NinferPrefillEpilogueMinRows = 65;

    /// <summary>Token tile for <paramref name="m"/> activation rows. RTX 5090 TTFT (574 / 930 /
    /// 1,950-token prompts): 128-token tiles 0.175 / 0.107 / 0.217 s, 256-token tiles
    /// 0.186 / 0.105 / 0.203 s; at 2,048-row chunks 256 is 7% faster.</summary>
    internal static int NinferTmaTokenTile(int m) => m >= 768 ? 256 : 128;

    internal bool NinferPrefillEpilogueAvailable
    {
        get
        {
            if (_niTmaState == 0)
            {
                _niTmaState = -1;
                try
                {
                    for (int b = 0; b < 2; b++)
                    {
                        _niTma[b] = new IntPtr[NinferTmaEpilogueCount];
                        for (int e = 0; e < NinferTmaEpilogueCount; e++)
                            _niTma[b][e] = module.GetFunction($"ts_q35_gemm_w4a4_tma{(b == 0 ? 256 : 128)}_ni_{NinferTmaEpilogues[e]}");
                    }
                    CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(16)).ThrowOnError();
                    try
                    {
                        IntPtr p = probe;
                        void** args = stackalloc void*[] { &p };
                        Launch(module.GetFunction("ts_q35_gemm_tma_smem_bytes"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
                        int* host = stackalloc int[3];
                        CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(12)).ThrowOnError();
                        // 256-token tiles: plain/round 3 stages, SwiGLU 2; 128-token tiles: 3 stages.
                        for (int e = 0; e < NinferTmaEpilogueCount; e++)
                        {
                            _niTmaSmem[e] = e < 2 ? host[0] : host[1];
                            _niTmaSmem[NinferTmaEpilogueCount + e] = host[2];
                        }
                    }
                    finally
                    {
                        CudaDriverApi.cuMemFree(probe);
                    }
                    for (int b = 0; b < 2; b++)
                        for (int e = 0; e < NinferTmaEpilogueCount; e++)
                            CudaDriverApi.cuFuncSetAttribute(_niTma[b][e], CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                _niTmaSmem[b * NinferTmaEpilogueCount + e]).ThrowOnError();
                    _niTmaState = 1;
                }
                catch (CudaException error) when (error.ErrorCode == 500)
                {
                    Console.Error.WriteLine("[q35-cuda] NInfer TMA prefill GEMMs absent from the main CUDA image; using separate round/SwiGLU/quantize passes");
                }
            }
            return _niTmaState > 0;
        }
    }

    /// <summary>Whether a projection fits the TMA tiles: K in whole 128-value tiles, outputs in
    /// 16-row pairs of 8 gate and 8 up rows (and float2 stores), 16-byte aligned planes.</summary>
    internal static bool NinferPrefillEpilogueFits(IntPtr wCodes, IntPtr wScales, int outDim, int k)
        => k > 0 && k % 128 == 0 && outDim > 0 && outDim % 16 == 0 && ((wCodes | wScales) & 15) == 0;

    /// <summary>y = epilogue(alpha * rowScale * (x W^T)) for m &gt;= 65 NVFP4 activation rows.
    /// For the SwiGLU modes wCodes are the interleaved gate|up planes and y is [m][outDim / 2].</summary>
    internal void GemmW4A4NinferPrefill(IntPtr wCodes, IntPtr wScales, int outDim, int k, IntPtr xCodes, IntPtr xScales,
        IntPtr rowScale, int m, float alpha, IntPtr y, NinferPrefillEpilogue epilogue, IntPtr stream)
        => LaunchNinferTma((int)epilogue, wCodes, wScales, outDim, k, xCodes, xScales, rowScale, m, alpha, y,
            IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 0f, stream);

    /// <summary>gate|up projection whose epilogue computes the BF16 SwiGLU activation and
    /// quantizes it to NVFP4 (codes, E4M3 group scales, row scale 1) with the down
    /// projection's input divisor, exactly as ts_ni_quant_nvfp4 would from the stored activation.</summary>
    internal void GemmW4A4NinferSwiGluQuant(IntPtr wCodes, IntPtr wScales, int outDim, int k, IntPtr xCodes, IntPtr xScales,
        IntPtr rowScale, int m, float alpha, bool roundedInputs, IntPtr qCodes, IntPtr qScales, IntPtr qRowScale,
        float qInputDivisor, IntPtr stream)
    {
        if (outDim % 256 != 0 || !(qInputDivisor > 0) || !float.IsFinite(qInputDivisor))
            throw new ArgumentOutOfRangeException(nameof(outDim));
        LaunchNinferTma(roundedInputs ? SwiGluQuantRoundedInputs : SwiGluQuant, wCodes, wScales, outDim, k, xCodes, xScales,
            rowScale, m, alpha, IntPtr.Zero, qCodes, qScales, qRowScale, qInputDivisor, stream);
    }

    private static void EncodeTma(byte* map, IntPtr address, long columns, long rows, uint boxColumns, uint boxRows, int swizzle)
    {
        ulong* dims = stackalloc ulong[2] { (ulong)columns, (ulong)rows };
        ulong* strides = stackalloc ulong[1] { (ulong)columns };
        uint* box = stackalloc uint[2] { boxColumns, boxRows };
        uint* elementStrides = stackalloc uint[2] { 1, 1 };
        CudaDriverApi.cuTensorMapEncodeTiled(map, 0, 2, address, dims, strides, box, elementStrides, 0, swizzle, 0, 0).ThrowOnError();
    }

    private void LaunchNinferTma(int epilogue, IntPtr wCodes, IntPtr wScales, int outDim, int k, IntPtr xCodes, IntPtr xScales,
        IntPtr rowScale, int m, float alpha, IntPtr y, IntPtr qCodes, IntPtr qScales, IntPtr qRowScale, float qInputDivisor, IntPtr stream)
    {
        if (!NinferPrefillEpilogueAvailable)
            throw new InvalidOperationException("The NInfer TMA prefill GEMMs are not in the loaded CUDA image.");
        if (m < NinferPrefillEpilogueMinRows)
            throw new ArgumentOutOfRangeException(nameof(m), "NInfer prefill epilogues serve prefill row counts only.");
        if (!NinferPrefillEpilogueFits(wCodes, wScales, outDim, k) || ((xCodes | xScales) & 15) != 0)
            throw new ArgumentException("The projection does not fit the TMA tiles; check NinferPrefillEpilogueFits first.");
        _gemmWorkSubmitted = true;
        int bm = NinferTmaTokenTile(m), b = bm == 256 ? 0 : 1;
        // CUtensorMap is 128 bytes and 64-byte aligned: xCodes, xScales, wCodes, wScales
        // (codes in 64-byte boxes with the 64-byte swizzle, scales in 16-byte boxes).
        byte* raw = stackalloc byte[4 * 128 + 64];
        byte* maps = (byte*)(((nint)raw + 63) & ~(nint)63);
        EncodeTma(maps, xCodes, k / 2, m, 64, (uint)bm, 2);
        EncodeTma(maps + 128, xScales, k / 16, m, 16, (uint)bm, 0);
        EncodeTma(maps + 256, wCodes, k / 2, outDim, 64, 128, 2);
        EncodeTma(maps + 384, wScales, k / 16, outDim, 16, 128, 0);
        IntPtr a3 = rowScale, a6 = y, a7 = qCodes, a8 = qScales, a9 = qRowScale;
        int a1 = outDim, a2 = k, a4 = m; float a5 = alpha, a10 = qInputDivisor;
        void** args = stackalloc void*[] { maps, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
        Launch(_niTma[b][epilogue], (uint)((m + bm - 1) / bm), (uint)((outDim + 127) / 128), 1, bm == 256 ? 384 : 288,
            _niTmaSmem[b * NinferTmaEpilogueCount + epilogue], stream, args);
        NinferPrefillEpilogueSubmissions++;
    }
}
