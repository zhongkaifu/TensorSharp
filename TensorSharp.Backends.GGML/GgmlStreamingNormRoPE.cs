// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public partial class GgmlBasicOps
{
    /// <summary>Apply per-head RMSNorm and NeoX RoPE in one CUDA graph, preserving
    /// the resident graph's fusion. Input/output are [tokens, heads*dim]; aliasing
    /// is supported. The scoped activation arena is outside the weight budget.</summary>
    public static void StreamingNormRoPE(Tensor input, Tensor norm, Tensor freqFactors, Tensor output,
        int heads, int headDim, int tokens, int startPos, int ropeDims, float eps, float freqBase)
    {
        if (heads <= 0 || headDim <= 0 || headDim % 2 != 0 || heads > int.MaxValue / headDim
            || tokens <= 0 || tokens > 65535 || startPos < 0 || startPos > int.MaxValue - tokens
            || ropeDims <= 0 || ropeDims > headDim || ropeDims % 2 != 0
            || !float.IsFinite(eps) || eps <= 0 || !float.IsFinite(freqBase) || freqBase <= 0)
            throw new ArgumentException("Invalid streaming norm/RoPE dimensions, positions or parameters.");
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(norm);
        ArgumentNullException.ThrowIfNull(output);
        if (input.Storage is not GgmlStorage inputStorage || inputStorage.Context.BackendType != GgmlBackendType.Cuda
            || inputStorage.Context.Degree != 1)
            throw new NotSupportedException("Streaming norm/RoPE requires one GGML CUDA rank.");
        foreach (var tensor in new[] { input, norm, freqFactors, output })
        {
            if (tensor == null) continue;
            if (tensor.Storage is not GgmlStorage storage || storage.Context != inputStorage.Context
                || storage.DeviceId != inputStorage.DeviceId || !tensor.IsContiguous() || tensor.ElementType != DType.Float32)
                throw new ArgumentException("Streaming norm/RoPE requires contiguous Float32 tensors on the same GGML CUDA context/device.");
        }
        if (input.DimensionCount != 2 || input.Sizes[0] != tokens || input.Sizes[1] != (long)heads * headDim
            || output.DimensionCount != 2 || output.Sizes[0] != tokens || output.Sizes[1] != (long)heads * headDim
            || norm.DimensionCount != 1 || norm.Sizes[0] != headDim
            || (freqFactors != null && (freqFactors.DimensionCount != 1 || freqFactors.Sizes[0] < ropeDims / 2
                || freqFactors.Sizes[0] > int.MaxValue)))
            throw new ArgumentException("Streaming norm/RoPE tensor shapes do not match its geometry.");
        input.Storage.EnsureHostReadable();
        norm.Storage.EnsureHostReadable();
        freqFactors?.Storage.EnsureHostReadable();
        GgmlNative.StreamingNormRoPE(GetBufferStart(input), GetBufferStart(norm),
            freqFactors == null ? IntPtr.Zero : GetBufferStart(freqFactors), GetBufferStart(output),
            heads, headDim, tokens, startPos, ropeDims, freqFactors == null ? 0 : (int)freqFactors.Sizes[0], eps, freqBase);
        InvalidateHostBuffer(GetBufferStart(output));
    }
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_StreamingNormRoPE(IntPtr input, IntPtr norm, IntPtr freqFactors, IntPtr output,
        int heads, int headDim, int tokens, int startPos, int ropeDims, int freqCount, float eps, float freqBase);

    internal static void StreamingNormRoPE(IntPtr input, IntPtr norm, IntPtr freqFactors, IntPtr output,
        int heads, int headDim, int tokens, int startPos, int ropeDims, int freqCount, float eps, float freqBase)
        => CheckResult(TSGgml_StreamingNormRoPE(input, norm, freqFactors, output,
            heads, headDim, tokens, startPos, ropeDims, freqCount, eps, freqBase), "streaming_norm_rope");
}
