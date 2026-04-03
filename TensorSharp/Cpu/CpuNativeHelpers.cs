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

namespace TensorSharp.Cpu
{
    public static class CpuNativeHelpers
    {
        public static IntPtr GetBufferStart(Tensor tensor)
        {
            IntPtr buffer = ((CpuStorage)tensor.Storage).buffer;
            return PtrAdd(buffer, tensor.StorageOffset * tensor.ElementType.Size());
        }

        private static IntPtr PtrAdd(IntPtr ptr, long offset)
        {
            return new IntPtr(ptr.ToInt64() + offset);
        }

    }
}
