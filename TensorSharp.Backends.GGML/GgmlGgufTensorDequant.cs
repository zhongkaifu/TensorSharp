// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

/// <summary>
/// GGUF tensor payload size (per GGML row layout) and dequantization to FP32 using the same
/// <c>ggml_type_traits::to_float</c> path as GGML CPU (Q4_K, Q8_0, IQ*, etc.).
/// </summary>
public static class GgmlGgufTensorDequant
{
    /// <summary>Bytes for one GGML row along <c>ne0</c>.</summary>
    public static long GetRowSizeBytes(int ggmlType, long ne0)
    {
        long rowBytes = GgmlNative.RowSizeBytesOrZero(ggmlType, ne0);
        if (rowBytes <= 0)
        {
            throw new NotSupportedException(
                $"GGML tensor type {ggmlType} is not supported, or ne[0]={ne0} is not aligned to the type block size.");
        }

        return rowBytes;
    }

    /// <summary>Total on-disk bytes for a contiguous tensor: <c>ggml_row_size(type, ne0) × ne1 × ne2 × ne3</c>.</summary>
    public static long GetTensorDataBytes(int ggmlType, long ne0, long ne1, long ne2, long ne3)
    {
        if (ne0 <= 0 || ne1 <= 0 || ne2 <= 0 || ne3 <= 0)
        {
            throw new ArgumentException("Tensor shape dimensions must be positive.");
        }

        long rowBytes = GetRowSizeBytes(ggmlType, ne0);
        return checked(rowBytes * ne1 * ne2 * ne3);
    }

    /// <summary>Decodes raw GGUF tensor bytes to FP32 (F32 copy, F16/BF16, or quantized).</summary>
    public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
    {
        GgmlNative.DequantizeGgufTensorToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
    }

    /// <summary>Decodes raw GGUF tensor bytes in unmanaged memory to FP32 in a managed array.</summary>
    public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
    {
        if (src == IntPtr.Zero)
        {
            throw new ArgumentException("Source pointer must be non-zero.", nameof(src));
        }

        if (numElements < 0 || numElements > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(numElements));
        }

        int n = (int)numElements;
        if (dstOffset < 0 || checked(dstOffset + n) > dst.Length)
        {
            throw new ArgumentException("Invalid destination range for dequantization.");
        }

        GCHandle hDst = GCHandle.Alloc(dst, GCHandleType.Pinned);
        try
        {
            IntPtr pDst = IntPtr.Add(hDst.AddrOfPinnedObject(), dstOffset * sizeof(float));
            GgmlNative.DequantizeGgufTensorToFloat32Native(ggmlType, src, pDst, numElements);
        }
        finally
        {
            if (hDst.IsAllocated)
            {
                hDst.Free();
            }
        }
    }

    /// <summary>Decodes raw GGUF tensor bytes in unmanaged memory to FP32 in unmanaged memory.</summary>
    public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
    {
        GgmlNative.DequantizeGgufTensorToFloat32Native(ggmlType, src, dst, numElements);
    }
}
