// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
﻿using System;

namespace TensorSharp.Core
{
    public static class TensorResultBuilder
    {
        // If a maybeResult is null, a new tensor will be constructed using the device id and element type of newTemplate
        public static Tensor GetWriteTarget(Tensor maybeResult, Tensor newTemplate, bool requireContiguous, params long[] requiredSizes)
        {
            return GetWriteTarget(maybeResult, newTemplate.Allocator, newTemplate.ElementType, requireContiguous, requiredSizes);
        }

        public static Tensor GetWriteTarget(Tensor maybeResult, IAllocator allocatorForNew, DType elementTypeForNew, bool requireContiguous, params long[] requiredSizes)
        {
            if (maybeResult != null)
            {
                if (!MatchesRequirements(maybeResult, requireContiguous, requiredSizes))
                {
                    string message = string.Format("output tensor does not match requirements. Tensor must have sizes {0}{1}",
                        string.Join(", ", requiredSizes),
                        requireContiguous ? "; and must be contiguous. " : ". ");

                    message += $"Tensor's actual shape is '{string.Join(", ", maybeResult.Sizes)}' and contiguous = '{maybeResult.IsContiguous()}'";

                    throw new InvalidOperationException(message);
                }
                return maybeResult;
            }
            else
            {
                return new Tensor(allocatorForNew, elementTypeForNew, requiredSizes);
            }
        }

        private static bool MatchesRequirements(Tensor tensor, bool requireContiguous, params long[] requiredSizes)
        {
            if (requireContiguous && !tensor.IsContiguous())
            {
                return false;
            }

            return ArrayEqual(tensor.Sizes, requiredSizes);
        }

        public static bool ArrayEqual<T>(T[] a, T[] b)
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (!a[i].Equals(b[i]))
                {
                    return false;
                }
            }

            return true;
        }

        public static bool ArrayEqualExcept<T>(T[] a, T[] b, int ignoreIndex)
        {
            if (a.Length != b.Length)
            {
                return false;
            }

            for (int i = 0; i < a.Length; ++i)
            {
                if (i == ignoreIndex)
                {
                    continue;
                }

                if (!a[i].Equals(b[i]))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
