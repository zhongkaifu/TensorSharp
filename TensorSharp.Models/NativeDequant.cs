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
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    internal static class NativeDequant
    {
        public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
            }
        }

        public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
            }
        }

        public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            try
            {
                GgmlGgufTensorDequant.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                ManagedQuantizedOps.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
            }
        }

        public static long RowSize(int ggmlType, long ne)
        {
            try
            {
                return GgmlGgufTensorDequant.GetRowSizeBytes(ggmlType, ne);
            }
            catch (Exception ex) when (ShouldUseManagedFallback(ex))
            {
                var type = (GgmlTensorType)ggmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                if (ne % blockSize != 0)
                    throw new NotSupportedException($"Tensor type {type} requires row length aligned to {blockSize}, got {ne}.");

                return (ne / blockSize) * GgufFile.GetTypeSize(type);
            }
        }

        private static bool ShouldUseManagedFallback(Exception ex)
        {
            if (ex is DllNotFoundException or EntryPointNotFoundException)
                return true;

            if (ex is TypeInitializationException tie && tie.InnerException != null)
                return ShouldUseManagedFallback(tie.InnerException);

            return false;
        }
    }
}

