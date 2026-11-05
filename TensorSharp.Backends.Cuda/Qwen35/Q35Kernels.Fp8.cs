// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>FP8 E4M3 row-scaled projections (NInfer v3 fp8_e4m3fn_row_bf16): the small-T BF16
/// tensor-core route for decode, verification and the drafter head (activations never quantized),
/// and for prefill the W8A8 TMA GEMM over per-row E4M3 activation codes (NInfer's A8 numerics:
/// scale = amax / 448). See tensorsharp_q35_kernels.cu and tensorsharp_small_t_mma.cuh.</summary>
internal sealed unsafe partial class Q35Kernels
{
    internal enum Fp8Epilogue { Plain = 0, Round = 1, SwiGlu = 2 }
    internal enum Fp8TmaEpilogue { Plain = 0, Round = 1, SwiGluRoundedInputs = 2, SwiGlu = 3 }
    private static readonly int[] Fp8Tiles = { 8, 16, 32 };
    private static readonly string[] Fp8SmallEpilogues = { "plain", "round", "swiglu" };
    private static readonly string[] Fp8TmaEpilogues = { "plain", "round", "swiglu_rin", "swiglu" };
    // [tile index][epilogue], F32 outputs: tile 0 reads F32 activations, tiles 1-2 BF16 activations.
    // BF16 in/out (drafter head): plain only.
    private readonly IntPtr[,] _fp8Small = new IntPtr[3, 3];
    private readonly IntPtr[] _fp8SmallBf16 = new IntPtr[3];
    private readonly IntPtr[,] _fp8Tma = new IntPtr[2, 4];
    private readonly int[] _fp8TmaSmem = new int[2];
    // T = 1 GEMV: [schedule b|c][epilogue], see TS_Q35_FP8_GEMV in tensorsharp_q35_kernels.cu.
    private readonly IntPtr[,] _fp8Gemv = new IntPtr[2, 3];
    private IntPtr _fp8Quant, _fp8Embed;
    private int _fp8State; // 0 unresolved, 1 available, -1 absent

    /// <summary>Activation rows from which a projection takes the W8A8 TMA GEMM (prefill).</summary>
    internal const int Fp8TmaMinRows = 65;
    /// <summary>Activation rows one small-T launch covers per grid row.</summary>
    internal const int Fp8SmallMaxTile = 32;

    internal bool Fp8Available
    {
        get
        {
            if (_fp8State != 0) return _fp8State > 0;
            _fp8State = -1;
            try
            {
                for (int t = 0; t < Fp8Tiles.Length; t++)
                {
                    for (int e = 0; e < Fp8SmallEpilogues.Length; e++)
                        _fp8Small[t, e] = module.GetFunction($"ts_q35_fp8_{(t == 0 ? "f32" : "bx")}_t{Fp8Tiles[t]}_{Fp8SmallEpilogues[e]}");
                    _fp8SmallBf16[t] = module.GetFunction($"ts_q35_fp8_bf16_t{Fp8Tiles[t]}_plain");
                }
                for (int b = 0; b < 2; b++)
                    for (int e = 0; e < Fp8TmaEpilogues.Length; e++)
                        _fp8Tma[b, e] = module.GetFunction($"ts_q35_gemm_w8a8_tma{(b == 0 ? 256 : 128)}_{Fp8TmaEpilogues[e]}");
                for (int g = 0; g < 2; g++)
                    for (int e = 0; e < Fp8SmallEpilogues.Length; e++)
                        _fp8Gemv[g, e] = module.GetFunction($"ts_q35_fp8_gemv_{(g == 0 ? "b" : "c")}_{Fp8SmallEpilogues[e]}");
                _fp8Quant = module.GetFunction("ts_q35_fp8_quant_act");
                _fp8Embed = module.GetFunction("ts_q35_embed_fp8");
                CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(16)).ThrowOnError();
                try
                {
                    IntPtr p = probe;
                    void** args = stackalloc void*[] { &p };
                    Launch(module.GetFunction("ts_q35_gemm_w8a8_tma_smem_bytes"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
                    int* host = stackalloc int[2];
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(8)).ThrowOnError();
                    _fp8TmaSmem[0] = host[0]; _fp8TmaSmem[1] = host[1];
                }
                finally { CudaDriverApi.cuMemFree(probe); }
                for (int b = 0; b < 2; b++)
                    for (int e = 0; e < Fp8TmaEpilogues.Length; e++)
                        CudaDriverApi.cuFuncSetAttribute(_fp8Tma[b, e], CudaDriverApi.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _fp8TmaSmem[b]).ThrowOnError();
                _fp8State = 1;
            }
            catch (CudaException error) when (error.ErrorCode == 500)
            {
                Console.Error.WriteLine("[q35-cuda] FP8 projection kernels absent from the main CUDA image");
            }
            return _fp8State > 0;
        }
    }

    private void RequireFp8()
    {
        if (!Fp8Available) throw new NotSupportedException("The main CUDA image has no FP8 projection kernels; rebuild the TensorSharp CUDA kernels.");
    }

    private static int Fp8TileIndex(int rows) => rows <= 8 ? 0 : rows <= 16 ? 1 : 2;

    /// <summary>Threads, activation rows and weight rows of one CTA of each instance
    /// (tensorsharp_q35_kernels.cu TS_Q35_FP8_SMALL: KW warps, TT rows, RT 16-row tiles).</summary>
    private static (int Threads, int Rows, int WeightRows) Fp8SmallGeometry(int tile)
        => tile switch { 0 => (128, 8, 16), 1 => (128, 16, 32), _ => (64, 32, 64) };

    /// <summary>True when a projection of <paramref name="rows"/> activation rows reads BF16 activations.</summary>
    internal static bool Fp8SmallTakesBf16(int rows) => Fp8TileIndex(rows) != 0;

    /// <summary>y[rows][ldy] = epilogue(rowScale * (x W^T)) on the small-T route, F32 outputs.
    /// x is F32 for up to 8 rows and BF16 above (<see cref="Fp8SmallTakesBf16"/>).
    /// SwiGlu: w is gate|up interleaved by 8 rows, y is [rows][n / 2].</summary>
    internal void ProjectFp8(IntPtr w, IntPtr scales, int n, int k, IntPtr x, int rows, IntPtr y, Fp8Epilogue epilogue, IntPtr stream)
    {
        RequireFp8();
        if (rows == 1 && TryGemvFp8(w, scales, n, k, x, y, epilogue, stream)) return;
        int tile = Fp8TileIndex(rows);
        var (threads, tt, weightRows) = Fp8SmallGeometry(tile);
        if (n % weightRows != 0 || k % 256 != 0 || rows <= 0) throw new ArgumentOutOfRangeException(nameof(n), "FP8 small-T projections need whole weight-row tiles and k % 256 == 0.");
        _gemmWorkSubmitted = true;
        int ldy = epilogue == Fp8Epilogue.SwiGlu ? n / 2 : n;
        IntPtr a0 = w, a1 = scales, a2 = x, a3 = y; int a4 = n, a5 = k, a6 = rows, a7 = ldy;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
        Launch(_fp8Small[tile, (int)epilogue], (uint)(n / weightRows), (uint)((rows + tt - 1) / tt), 1, threads, 0, stream, args);
    }

    /// <summary>False when TS_Q35_T1_GEMV=off: one-row projections then take the small-T tensor-core
    /// tile (FP8) and the legacy A16 kernel (NVFP4), for A/B measurements.</summary>
    internal static readonly bool T1GemvEnabled = !string.Equals(
        Environment.GetEnvironmentVariable("TS_Q35_T1_GEMV")?.Trim(), "off", StringComparison.OrdinalIgnoreCase);

    /// <summary>One activation row on the CUDA cores: schedule b (4 warps x 4 rows x 16 codes) for the
    /// 16384-row GDN input projection, c (8 warps x 2 rows x 16 codes) otherwise; both cover 16 weight
    /// rows a CTA. Measured on the RTX 5090: 1.60-1.70 TB/s against 1.45-1.60 for the 8-row MMA tile.</summary>
    private bool TryGemvFp8(IntPtr w, IntPtr scales, int n, int k, IntPtr x, IntPtr y, Fp8Epilogue epilogue, IntPtr stream)
    {
        if (!T1GemvEnabled || n % 16 != 0 || k % 512 != 0 || ((long)x & 15) != 0) return false;
        int g = n == 16384 && epilogue != Fp8Epilogue.SwiGlu ? 0 : 1;
        _gemmWorkSubmitted = true;
        IntPtr a0 = w, a1 = scales, a2 = x, a3 = y; int a4 = k;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4 };
        Launch(_fp8Gemv[g, (int)epilogue], (uint)(n / 16), 1, 1, g == 0 ? 128 : 256, 0, stream, args);
        return true;
    }

    /// <summary>BF16 activations to BF16 outputs (the drafter's proposals over the target head).</summary>
    internal void ProjectFp8Bf16(IntPtr w, IntPtr scales, int n, int k, IntPtr x, int rows, IntPtr y, IntPtr stream)
    {
        RequireFp8();
        int tile = Fp8TileIndex(rows);
        var (threads, tt, weightRows) = Fp8SmallGeometry(tile);
        if (n % weightRows != 0 || k % 256 != 0 || rows <= 0) throw new ArgumentOutOfRangeException(nameof(n));
        IntPtr a0 = w, a1 = scales, a2 = x, a3 = y; int a4 = n, a5 = k, a6 = rows, a7 = n;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7 };
        Launch(_fp8SmallBf16[tile], (uint)(n / weightRows), (uint)((rows + tt - 1) / tt), 1, threads, 0, stream, args);
    }

    /// <summary>Per-row E4M3 codes [rows][k] and F32 scales [rows] of F32 activations.</summary>
    internal void QuantizeFp8Activations(IntPtr x, IntPtr codes, IntPtr scales, int k, int rows, IntPtr stream)
    {
        RequireFp8();
        if (k % 4 != 0 || rows <= 0) throw new ArgumentOutOfRangeException(nameof(k));
        IntPtr a0 = x, a1 = codes, a2 = scales; int a3 = k;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3 };
        Launch(_fp8Quant, (uint)rows, 1, 1, 256, 0, stream, args);
    }

    internal static bool Fp8TmaFits(IntPtr w, IntPtr xCodes, int n, int k)
        => k > 0 && k % 64 == 0 && n > 0 && n % 16 == 0 && ((w | xCodes) & 15) == 0;

    /// <summary>y = epilogue(acc * xScale[token] * wScale[row]) for prefill row counts.</summary>
    internal void GemmW8A8(IntPtr w, IntPtr wScales, int n, int k, IntPtr xCodes, IntPtr xScales, int m, IntPtr y,
        Fp8TmaEpilogue epilogue, IntPtr stream)
    {
        RequireFp8();
        if (m < Fp8TmaMinRows || !Fp8TmaFits(w, xCodes, n, k)) throw new ArgumentException("The projection does not fit the W8A8 TMA tiles.");
        _gemmWorkSubmitted = true;
        int bm = NinferTmaTokenTile(m), b = bm == 256 ? 0 : 1;
        byte* raw = stackalloc byte[4 * 128 + 64];
        byte* maps = (byte*)(((nint)raw + 63) & ~(nint)63);
        new Span<byte>(maps, 4 * 128).Clear();
        EncodeTma(maps, xCodes, k, m, 64, (uint)bm, 2);
        EncodeTma(maps + 256, w, k, n, 64, 128, 2);
        IntPtr a3 = xScales, a4 = wScales, a6 = y;
        int a1 = n, a2 = k, a5 = m;
        void** args = stackalloc void*[] { maps, &a1, &a2, &a3, &a4, &a5, &a6 };
        Launch(_fp8Tma[b, (int)epilogue], (uint)((m + bm - 1) / bm), (uint)((n + 127) / 128), 1, bm == 256 ? 384 : 288,
            _fp8TmaSmem[b], stream, args);
    }

    internal void EmbedFp8(IntPtr codes, IntPtr scales, IntPtr ids, IntPtr output, int rows, int width, int vocabulary, IntPtr stream)
    {
        RequireFp8();
        IntPtr a0 = codes, a1 = scales, a2 = ids, a3 = output; int a4 = width, a5 = vocabulary;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5 };
        Launch(_fp8Embed, (uint)rows, 1, 1, 256, 0, stream, args);
    }
}
