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

namespace TensorSharp
{
    public class TensorDimIterState
    {
        private readonly ReadOnlyMemory<long> sizes;
        private readonly ReadOnlyMemory<long> strides;
        private readonly int dimensionCount;
        private readonly int iterationDim;
        private readonly long[] counter;

        public long stride, size;
        unsafe public float* data;

        unsafe public TensorDimIterState(float* buffer, int dimCount, long[] sizes, long[] strides, int iterationDim)
            : this(buffer, dimCount, (ReadOnlyMemory<long>)sizes, (ReadOnlyMemory<long>)strides, iterationDim)
        {
        }

        unsafe public TensorDimIterState(float* buffer, int dimCount, ReadOnlyMemory<long> sizes, ReadOnlyMemory<long> strides, int iterationDim)
        {
            if (sizes.Length < dimCount || strides.Length < dimCount)
            {
                throw new ArgumentException("sizes and strides must contain dimCount elements");
            }

            this.sizes = sizes;
            this.strides = strides;
            this.iterationDim = iterationDim;
            dimensionCount = dimCount;

            ReadOnlySpan<long> sizesSpan = sizes.Span;
            ReadOnlySpan<long> stridesSpan = strides.Span;

            data = buffer;

            size = sizesSpan[iterationDim];
            stride = stridesSpan[iterationDim];

            counter = new long[dimCount];
            for (int i = 0; i < dimCount; ++i)
            {
                counter[i] = 0;
            }
        }

        // Returns true if there is another block to iterate over,
        // returns false if we are at end of iteration
        unsafe public bool NextBlock()
        {
            if (dimensionCount == 1)
            {
                return false;
            }

            ReadOnlySpan<long> sizesSpan = sizes.Span;
            ReadOnlySpan<long> stridesSpan = strides.Span;

            for (int i = 0; i < dimensionCount; ++i)
            {
                if (i == iterationDim)
                {
                    if (i == dimensionCount - 1)
                    {
                        return false;
                    }
                    continue;
                }

                counter[i]++;
                data += stridesSpan[i];

                if (counter[i] == sizesSpan[i])
                {
                    if (i == dimensionCount - 1)
                    {
                        return false;
                    }
                    else
                    {
                        data -= counter[i] * stridesSpan[i];
                        counter[i] = 0;
                    }
                }
                else
                {
                    break;
                }
            }

            return true;
        }
    }
}
