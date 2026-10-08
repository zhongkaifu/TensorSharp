// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public partial class GgmlBasicOps
{
    /// <summary>Explicit single-rank CUDA flash attention without a retained graph
    /// or materialized-attention fallback. Q is contiguous [heads, query, dim],
    /// K/V are [kvHeads, stride, dim], output is [query, heads*dim]. Only the
    /// leading valid KV rows are uploaded. This operation does not own weights;
    /// its temporary attention arena is outside the file-weight staging budget.</summary>
    public static void StreamingFlashAttention(Tensor q, Tensor k, Tensor v, Tensor output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen,
        int kvStride, int maskStartPos, int slidingWindow, float scale)
    {
        if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0 || headDim <= 0 || numHeads > int.MaxValue / headDim
            || seqLen <= 0 || kvLen < seqLen || kvStride < kvLen || maskStartPos < 0
            || maskStartPos > kvLen - seqLen || slidingWindow < 0 || !float.IsFinite(scale))
            throw new ArgumentException("Invalid streaming flash attention dimensions or mask.");
        ArgumentNullException.ThrowIfNull(q);
        ArgumentNullException.ThrowIfNull(k);
        ArgumentNullException.ThrowIfNull(v);
        ArgumentNullException.ThrowIfNull(output);
        if (q.Storage is not GgmlStorage qStorage || qStorage.Context.BackendType != GgmlBackendType.Cuda
            || qStorage.Context.Degree != 1)
            throw new NotSupportedException("Streaming flash attention requires one GGML CUDA rank.");
        foreach (var tensor in new[] { q, k, v, output })
            if (tensor.Storage is not GgmlStorage storage || storage.Context != qStorage.Context
                || storage.DeviceId != qStorage.DeviceId || !tensor.IsContiguous())
                throw new ArgumentException("Streaming flash attention requires contiguous tensors on the same GGML CUDA context/device.");
        if (q.ElementType != DType.Float32 || output.ElementType != DType.Float32
            || k.ElementType != v.ElementType || k.ElementType is not (DType.Float16 or DType.Float32)
            || q.DimensionCount != 3 || q.Sizes[0] != numHeads || q.Sizes[1] != seqLen || q.Sizes[2] != headDim
            || k.DimensionCount != 3 || k.Sizes[0] != numKvHeads || k.Sizes[1] != kvStride || k.Sizes[2] != headDim
            || v.DimensionCount != 3 || v.Sizes[0] != numKvHeads || v.Sizes[1] != kvStride || v.Sizes[2] != headDim
            || output.DimensionCount != 2 || output.Sizes[0] != seqLen || output.Sizes[1] != checked((long)numHeads * headDim))
            throw new ArgumentException("Streaming flash attention tensor shapes or dtypes do not match its declared geometry.");
        q.Storage.EnsureHostReadable();
        k.Storage.EnsureHostReadable();
        v.Storage.EnsureHostReadable();
        GgmlNative.StreamingFlashAttention(GetBufferStart(q), GetBufferStart(k), GetBufferStart(v), GetBufferStart(output),
            numHeads, numKvHeads, headDim, seqLen, kvLen, kvStride, maskStartPos, slidingWindow, scale,
            k.ElementType == DType.Float16 ? 1 : 0);
        // The scoped native graph writes the host output, so subsequent per-op
        // kernels must not reuse an older device copy of the same pool address.
        InvalidateHostBuffer(GetBufferStart(output));
    }
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_StreamingFlashAttention(IntPtr q, IntPtr k, IntPtr v, IntPtr output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen, int kvStride,
        int maskStartPos, int slidingWindow, float scale, int kvType);

    internal static void StreamingFlashAttention(IntPtr q, IntPtr k, IntPtr v, IntPtr output,
        int numHeads, int numKvHeads, int headDim, int seqLen, int kvLen, int kvStride,
        int maskStartPos, int slidingWindow, float scale, int kvType)
        => CheckResult(TSGgml_StreamingFlashAttention(q, k, v, output,
            numHeads, numKvHeads, headDim, seqLen, kvLen, kvStride, maskStartPos, slidingWindow, scale, kvType),
            "streaming_flash_attention");
}
