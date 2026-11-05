// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

/// <summary>8-bit prompt attention (tensorsharp_q35_attn8.cuh): the visible keys are converted once
/// per prompt chunk and layer into a band (INT8 K with a scale per key, E4M3 V^T with a power-of-two
/// scale per 32-key tile), then Q K^T runs on the INT8 tensor cores and P V on the block-scaled E4M3
/// ones. Ignis' FP8 prompt attention (after mkyt/qwen3-27b-5090) is the model. On by default for prompt
/// chunks with at least <see cref="Attn8MinKeys"/> visible keys; TS_Q35_ATTN8=0 keeps the 16-bit kernels.
/// RTX 5090, Qwen3.8-27B TTFT: GGUF route 32K 3.99 -> 3.65 s, 63K 9.58 -> 8.21 s; NInfer route 32K 3.94 ->
/// 3.43 s, 63K 10.0 -> 7.87 s; every 8K-63K retrieval/order check passed (2% relative RMS against the
/// F16 kernel per layer).</summary>
internal sealed unsafe partial class Q35Kernels
{
    internal static readonly bool Attn8Enabled = Environment.GetEnvironmentVariable("TS_Q35_ATTN8") != "0";
    /// <summary>Fewest visible keys for which a prompt chunk takes the 8-bit path.</summary>
    internal const int Attn8MinKeys = 1024;
    internal const int Attn8TileKeys = 32;
    /// <summary>Cache row formats the band reads: TensorSharp q8_0 (GGUF route), BF16 and FP8 (NInfer route).</summary>
    internal enum Attn8Cache { Q8 = 0, Bf16 = 1, Fp8 = 2 }
    private readonly IntPtr[] _attn8Bands = new IntPtr[3];
    private IntPtr _attn8Band, _attn8Prefill;   // _attn8Band: resolved marker
    private int _attn8BandSmem, _attn8PrefillSmem;

    /// <summary>Band bytes per KV head and key capacity: K8 rows, K scales, V^T tiles, V tile scales.</summary>
    internal static (long K, long KScales, long VT, long VScales) Attn8BandBytes(int keysCap, int nKv)
        => ((long)nKv * keysCap * AttnHeadDim, (long)nKv * keysCap * 4, (long)nKv * keysCap * AttnHeadDim, (long)nKv * (keysCap / Attn8TileKeys));

    private void RequireAttn8()
    {
        if (_attn8Band != IntPtr.Zero) return;
        _attn8Bands[0] = module.GetFunction("ts_q35_attn8_band_q8");
        _attn8Bands[1] = module.GetFunction("ts_q35_attn8_band_bf16");
        _attn8Bands[2] = module.GetFunction("ts_q35_attn8_band_fp8");
        _attn8Band = _attn8Bands[0];
        _attn8Prefill = module.GetFunction("ts_q35_attn8_prefill_f32");
        CudaDriverApi.cuMemAlloc(out IntPtr probe, new UIntPtr(16)).ThrowOnError();
        try
        {
            IntPtr p = probe;
            void** args = stackalloc void*[] { &p };
            Launch(module.GetFunction("ts_q35_attn8_smem_bytes"), 1, 1, 1, 1, 0, IntPtr.Zero, args);
            int* host = stackalloc int[2];
            CudaDriverApi.cuMemcpyDtoH((IntPtr)host, probe, new UIntPtr(8)).ThrowOnError();
            _attn8BandSmem = host[0];
            _attn8PrefillSmem = host[1];
        }
        finally { CudaDriverApi.cuMemFree(probe); }
        const int MaxDynamicShared = 8; // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
        foreach (IntPtr band in _attn8Bands)
            CudaDriverApi.cuFuncSetAttribute(band, MaxDynamicShared, _attn8BandSmem).ThrowOnError();
        CudaDriverApi.cuFuncSetAttribute(_attn8Prefill, MaxDynamicShared, _attn8PrefillSmem).ThrowOnError();
    }

    /// <summary>The band of keys [0, keys) of one sequence's caches (its row of block pointers).</summary>
    internal void Attn8Band(Attn8Cache cache, IntPtr blocks, long kOff, long vOff, int keys, int keysCap, int nKv,
        IntPtr k8, IntPtr kScales, IntPtr vt, IntPtr vScales, IntPtr stream)
    {
        RequireAttn8();
        if (keys < 1 || keys > keysCap || keysCap % Attn8TileKeys != 0) throw new ArgumentOutOfRangeException(nameof(keys));
        IntPtr a0 = blocks, a5 = k8, a6 = kScales, a7 = vt, a8 = vScales; long a1 = kOff, a2 = vOff; int a3 = keys, a4 = keysCap;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8 };
        Launch(_attn8Bands[(int)cache], CeilDiv(keys, Attn8TileKeys), (uint)nKv, 1, 256, _attn8BandSmem, stream, args);
    }

    /// <summary>Gated causal attention of one sequence's prompt rows over its band.</summary>
    internal void Attn8Prefill(IntPtr q, IntPtr gate, IntPtr keyEnd, IntPtr k8, IntPtr kScales, IntPtr vt, IntPtr vScales,
        int keysCap, int rowsPerSeq, int nHead, int nKv, float scale, IntPtr output, IntPtr stream)
    {
        RequireAttn8();
        if (nHead != 24 || nKv != 4) throw new NotSupportedException("The 8-bit prompt attention is built for 24 query heads over 4 KV heads.");
        IntPtr a0 = q, a1 = gate, a2 = keyEnd, a3 = k8, a4 = kScales, a5 = vt, a6 = vScales, a13 = output;
        int a7 = keysCap, a8 = rowsPerSeq, a9 = nHead, a10 = nKv; float a11 = scale;
        void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10, &a11, &a13 };
        uint tiles = CeilDiv((long)rowsPerSeq * (nHead / nKv), 128);
        Launch(_attn8Prefill, tiles, (uint)nKv, 1, 256, _attn8PrefillSmem, stream, args);
    }
}
