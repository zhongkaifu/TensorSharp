using System;

namespace TensorSharp.GGML;

/// <summary>
/// GGUF tensor payload size (per GGML row layout) and dequantization to FP32 using the same
/// <c>ggml_type_traits::to_float</c> path as GGML CPU (Q4_K, Q8_0, IQ*, etc.).
/// </summary>
public static class GgmlGgufTensorDequant
{
    /// <summary>Total on-disk bytes for a contiguous tensor: <c>ggml_row_size(type, ne0) × ne1 × ne2 × ne3</c>.</summary>
    public static long GetTensorDataBytes(int ggmlType, long ne0, long ne1, long ne2, long ne3)
    {
        if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0)
        {
            throw new ArgumentException("Tensor shape dimensions must be positive.");
        }

        long rowBytes = GgmlNative.RowSizeBytesOrZero(ggmlType, ne0);
        if (rowBytes <= 0)
        {
            throw new NotSupportedException(
                $"GGML tensor type {ggmlType} is not supported, or ne[0]={ne0} is not aligned to the type block size.");
        }

        return checked(rowBytes * ne1 * ne2 * ne3);
    }

    /// <summary>Decodes raw GGUF tensor bytes to FP32 (F32 copy, F16/BF16, or quantized).</summary>
    public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
    {
        GgmlNative.DequantizeGgufTensorToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
    }
}
