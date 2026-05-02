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
using System.Text;

namespace TensorSharp
{
    public static class TensorDimensionHelpers
    {
        public static long ElementCount(long[] sizes)
        {
            return ElementCount((ReadOnlySpan<long>)sizes);
        }

        public static long ElementCount(ReadOnlySpan<long> sizes)
        {
            if (sizes.Length == 0)
            {
                return 0;
            }

            long total = 1L;
            for (int i = 0; i < sizes.Length; ++i)
            {
                total *= sizes[i];
            }

            return total;
        }

        public static long GetStorageSize(long[] sizes, long[] strides)
        {
            return GetStorageSize((ReadOnlySpan<long>)sizes, (ReadOnlySpan<long>)strides);
        }

        public static long GetStorageSize(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides)
        {
            if (sizes.Length != strides.Length)
            {
                throw new ArgumentException("sizes and strides must have the same length");
            }

            long offset = 0;
            for (int i = 0; i < sizes.Length; ++i)
            {
                offset += (sizes[i] - 1) * strides[i];
            }
            return offset + 1; // +1 to count last element, which is at *index* equal to offset
        }

        // Returns the stride required for a tensor to be contiguous.
        // If a tensor is contiguous, then the elements in the last dimension are contiguous in memory,
        // with lower numbered dimensions having increasingly large strides.
        public static long[] GetContiguousStride(long[] dims)
        {
            return GetContiguousStride((ReadOnlySpan<long>)dims);
        }

        public static long[] GetContiguousStride(ReadOnlySpan<long> dims)
        {
            long acc = 1;
            long[] stride = new long[dims.Length];

            for (int i = dims.Length - 1; i >= 0; --i)
            {
                stride[i] = acc;
                acc *= dims[i];
            }

            //if (dims[dims.Length - 1] == 1)
            //{
            //    stride[dims.Length - 1] = 0;
            //}

            return stride;
        }

        public static string FormatDimensions(ReadOnlySpan<long> dims, string separator = " ")
        {
            if (dims.Length == 0)
            {
                return string.Empty;
            }

            StringBuilder builder = new StringBuilder();
            builder.Append(dims[0]);
            for (int i = 1; i < dims.Length; ++i)
            {
                builder.Append(separator).Append(dims[i]);
            }

            return builder.ToString();
        }
    }
}
