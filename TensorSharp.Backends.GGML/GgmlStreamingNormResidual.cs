// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

public partial class GgmlBasicOps
{
    /// <summary>Update residual with RMSNorm(input)*norm + residual in one CUDA
    /// graph, preserving the resident norm/mul/add fusion. The scoped activation
    /// arena is outside the weight budget; no pointers or graph are retained.</summary>
    public static void StreamingNormResidual(Tensor residual, Tensor input, Tensor norm, float eps)
    {
        if (!float.IsFinite(eps) || eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps));
        ArgumentNullException.ThrowIfNull(residual);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(norm);
        if (input.Storage is not GgmlStorage inputStorage || inputStorage.Context.BackendType != GgmlBackendType.Cuda
            || inputStorage.Context.Degree != 1)
            throw new NotSupportedException("Streaming norm/residual requires one GGML CUDA rank.");
        foreach (var tensor in new[] { input, norm, residual })
            if (tensor.Storage is not GgmlStorage storage || storage.Context != inputStorage.Context
                || storage.DeviceId != inputStorage.DeviceId || !tensor.IsContiguous() || tensor.ElementType != DType.Float32)
                throw new ArgumentException("Streaming norm/residual requires contiguous Float32 tensors on the same GGML CUDA context/device.");
        if (input.DimensionCount != 2 || input.Sizes[0] <= 0 || input.Sizes[0] > int.MaxValue
            || input.Sizes[1] <= 0 || input.Sizes[1] > int.MaxValue
            || residual.DimensionCount != 2 || residual.Sizes[0] != input.Sizes[0] || residual.Sizes[1] != input.Sizes[1]
            || norm.DimensionCount != 1 || norm.Sizes[0] != input.Sizes[1])
            throw new ArgumentException("Streaming norm/residual tensor shapes do not match.");
        input.Storage.EnsureHostReadable();
        residual.Storage.EnsureHostReadable();
        norm.Storage.EnsureHostReadable();
        GgmlNative.StreamingNormResidual(GetBufferStart(residual), GetBufferStart(input), GetBufferStart(norm),
            (int)input.Sizes[1], (int)input.Sizes[0], eps);
        InvalidateHostBuffer(GetBufferStart(residual));
    }
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
    private static partial int TSGgml_StreamingNormResidual(IntPtr residual, IntPtr input, IntPtr norm,
        int width, int rows, float eps);

    internal static void StreamingNormResidual(IntPtr residual, IntPtr input, IntPtr norm, int width, int rows, float eps)
        => CheckResult(TSGgml_StreamingNormResidual(residual, input, norm, width, rows, eps), "streaming_norm_residual");
}
