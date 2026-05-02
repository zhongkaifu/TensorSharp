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
    public class TensorIterState
    {
        private readonly ReadOnlyMemory<long> sizes;
        private readonly ReadOnlyMemory<long> strides;

        private long stride, size;
        private int dim;
        private readonly long[] counter;
        private readonly int step;

        private long index;
        unsafe public float* data;

        unsafe public TensorIterState(float* buffer, int dimCount, long[] sizes, long[] strides, int step = 1)
            : this(buffer, dimCount, (ReadOnlyMemory<long>)sizes, (ReadOnlyMemory<long>)strides, step)
        {
        }

        unsafe public TensorIterState(float* buffer, int dimCount, ReadOnlyMemory<long> sizes, ReadOnlyMemory<long> strides, int step = 1)
        {
            if (sizes.Length < dimCount || strides.Length < dimCount)
            {
                throw new ArgumentException("sizes and strides must contain dimCount elements");
            }

            this.sizes = sizes;
            this.strides = strides;
            this.step = step;

            ReadOnlySpan<long> sizesSpan = sizes.Span;
            ReadOnlySpan<long> stridesSpan = strides.Span;

            index = 0;
            data = buffer;

            for (dim = dimCount - 1; dim >= 0; dim--)
            {
                if (sizesSpan[dim] != 1)
                {
                    break;
                }
            }

            // Get stride for dimension
            stride = (dim == -1 ? 0 : stridesSpan[dim]);

            // Find largest contiguous section.
            // Note: this updates dim and size.
            size = 1;
            for (dim = dimCount - 1; dim >= 0; dim--)
            {
                if (stridesSpan[dim] == size)
                {
                    size *= sizesSpan[dim];
                }
                else
                {
                    break;
                }
            }

            if (size % step != 0)
            {
                throw new ArgumentException($"Size '{size}' mod step '{step}' must be zero.");
            }

            // Counter keeps track of dimensions outside the contiguous block.
            counter = new long[dim + 1];
            for (int i = 0; i < dim + 1; ++i)
            {
                counter[i] = 0;
            }
        }

        public bool ReachedBlockEnd()
        {
            return !(index < size);
        }

        public void BlockStep()
        {
            unsafe
            {
                index += step;
                data += (stride * step);
            }
        }

        // Returns true if there is another block to iterate over,
        // returns false if we are at end of iteration
        public bool NextBlock()
        {
            unsafe
            {
                // If not at end of current block yet, do nothing
                if (index == size)
                {
                    // If contiguous block encompassed all dimensions, we are done
                    if (dim == -1)
                    {
                        return false;
                    }

                    ReadOnlySpan<long> sizesSpan = sizes.Span;
                    ReadOnlySpan<long> stridesSpan = strides.Span;

                    // Reset data offset
                    data -= size * stride;

                    // Update counter and data for next contiguous block
                    for (long j = dim; j >= 0; --j)
                    {
                        counter[j]++;
                        data += stridesSpan[(int)j];

                        if (counter[j] == sizesSpan[(int)j])
                        {
                            if (j == 0)
                            {
                                return false;
                            }
                            else
                            {
                                data -= counter[j] * stridesSpan[(int)j];
                                counter[j] = 0;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    index = 0;
                }

                return true;
            }
        }
    }
}
