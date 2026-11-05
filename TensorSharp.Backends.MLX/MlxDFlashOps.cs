// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.MLX;

/// <summary>
/// The DFlash drafter's grouped block convolution on the MLX device. The op-by-op drafter ran it on
/// the host (ModelBase.DFlashGroupedConvolve), and reading the filter coefficients and the rows back
/// forced an MLX evaluation and a host wait twice per sublayer: four per layer, twenty per block
/// draft of the 5-layer Qwen3.8 DFlash2 drafter, which made a draft cost ~50 ms on MLX against
/// ~14 ms on ggml_metal (fused graph). Same arithmetic as the host loop, in the same order.
/// </summary>
public static class MlxDFlashOps
{
    /// <summary>
    /// result[r][c] = sum over taps t &lt;= r of (baseKernel[side][t][c] + coef[r][deltaOffset + t * groups + c / groupSize]) * x[r - t][c],
    /// tap 0 first, then each further tap added. x, result: [rows, hidden]; coef: [rows, coefStride];
    /// baseKernel: [2, taps, hidden]. False when a tensor is not on the MLX device.
    /// </summary>
    public static bool TryGroupedConvolve(Tensor result, Tensor x, Tensor coef, int coefStride, int deltaOffset, Tensor baseKernel,
        int side, int rows, int hidden, int taps, int groupSize, int groups)
    {
        if (!IsDeviceF32(result) || !IsDeviceF32(x) || !IsDeviceF32(coef) || !IsDeviceF32(baseKernel)
            || rows <= 0 || hidden <= 0 || taps <= 0 || groupSize <= 0 || groups * groupSize != hidden)
            return false;
        return MlxWorker.Shared.Invoke(() =>
        {
            MlxNative.MlxArray xv = default, cv = default, bv = default, output = default;
            try
            {
                xv = ((MlxStorage)x.Storage).CreateArrayView(x);
                cv = ((MlxStorage)coef.Storage).CreateArrayView(coef);
                bv = ((MlxStorage)baseKernel.Storage).CreateArrayView(baseKernel);
                output = MlxNative.DFlashGroupedConvolve(xv, cv, bv, rows, hidden, taps, groupSize, groups, coefStride, deltaOffset, side);
                var storage = (MlxStorage)result.Storage;
                if (result.StorageOffset == 0 && result.Storage.ElementCount == result.ElementCount())
                    storage.ReplaceDeviceArray(output);
                else
                {
                    storage.UpdateDeviceSlice(result, output);
                    MlxNative.FreeArray(output);
                }
                output = default;
                return true;
            }
            finally
            {
                MlxNative.FreeArray(xv); MlxNative.FreeArray(cv); MlxNative.FreeArray(bv); MlxNative.FreeArray(output);
            }
        });
    }

    private static bool IsDeviceF32(Tensor t) => t != null && t.Storage is MlxStorage && t.ElementType == DType.Float32 && t.IsContiguous();
}

internal static partial class MlxNative
{
    private static MlxFastMetalKernel dflashConvKernel;

    internal static MlxArray DFlashGroupedConvolve(MlxArray x, MlxArray coef, MlxArray baseKernel, int rows, int hidden, int taps,
        int groupSize, int groups, int coefStride, int deltaOffset, int side)
    {
        lock (fastKernelSync)
        {
            if (!dflashConvKernel.IsValid)
            {
                // Shapes and offsets are inputs, not template values: one compile for every call.
                dflashConvKernel = CreateFastMetalKernel("tensorsharp_dflash_grouped_conv",
                    new[] { "x", "coef", "base_kernel", "dims" }, new[] { "y" }, """
const uint c = thread_position_in_grid.x;
const uint r = thread_position_in_grid.y;
const int rows_ = dims[0], hidden = dims[1], taps = dims[2], group_size = dims[3], groups = dims[4];
const int coef_stride = dims[5], delta_offset = dims[6], side = dims[7];
if (int(c) >= hidden || int(r) >= rows_) {
    return;
}
const int g = int(c) / group_size;
float acc = 0.0f;
for (int t = 0; t < taps && t <= int(r); ++t) {
    const float bt = base_kernel[(side * taps + t) * hidden + int(c)];
    const float d = coef[int(r) * coef_stride + delta_offset + t * groups + g];
    const float v = (bt + d) * x[(int(r) - t) * hidden + int(c)];
    acc = t == 0 ? v : acc + v;
}
y[int(r) * hidden + int(c)] = acc;
""", string.Empty);
                if (!dflashConvKernel.IsValid)
                    throw new NotSupportedException("Unable to initialize the MLX DFlash grouped convolution kernel.");
            }
        }
        MlxFastMetalKernelConfig config = mlx_fast_metal_kernel_config_new();
        MlxVectorArray inputs = default, outputs = default;
        MlxArray dims = default;
        try
        {
            int[] dimValues = { rows, hidden, taps, groupSize, groups, coefStride, deltaOffset, side };
            unsafe
            {
                // mlx_array_new_data copies the host values.
                fixed (int* dp = dimValues)
                    dims = NewArrayFromHost((IntPtr)dp, new[] { dimValues.Length }, ToMlxDtype(DType.Int32));
            }
            int[] shape = { rows, hidden };
            Check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, (nuint)shape.Length, ToMlxDtype(DType.Float32)), "configuring DFlash conv output");
            Check(mlx_fast_metal_kernel_config_set_grid(config, hidden, rows, 1), "configuring DFlash conv grid");
            Check(mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1), "configuring DFlash conv threadgroup");
            inputs = CreateVectorArray(x, coef, baseKernel, dims);
            outputs = mlx_vector_array_new();
            Check(mlx_fast_metal_kernel_apply(ref outputs, dflashConvKernel, inputs, config, DefaultStream()), "running DFlash conv");
            Check(mlx_vector_array_get(out MlxArray result, outputs, 0), "reading DFlash conv output");
            return result;
        }
        finally
        {
            FreeArray(dims);
            if (inputs.IsValid) _ = mlx_vector_array_free(inputs);
            if (outputs.IsValid) _ = mlx_vector_array_free(outputs);
            if (config.IsValid) _ = mlx_fast_metal_kernel_config_free(config);
        }
    }
}
