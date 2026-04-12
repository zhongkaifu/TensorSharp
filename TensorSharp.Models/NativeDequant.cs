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

namespace TensorSharp.Models
{
    internal static class NativeDequant
    {
        public static void DequantizeToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, srcOffset, dst, dstOffset, numElements);
        }

        public static void DequantizeToFloat32(int ggmlType, IntPtr src, float[] dst, int dstOffset, long numElements)
        {
            ManagedQuantizedOps.DequantizeToFloat32(ggmlType, src, dst, dstOffset, numElements);
        }

        public static void DequantizeToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            ManagedQuantizedOps.DequantizeToFloat32Native(ggmlType, src, dst, numElements);
        }

        public static long RowSize(int ggmlType, long ne)
        {
            return ManagedQuantizedOps.RowSize(ggmlType, ne);
        }
    }
}

