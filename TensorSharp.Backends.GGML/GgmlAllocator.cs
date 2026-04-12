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

namespace TensorSharp.GGML
{
    [Serializable]
    public class GgmlAllocator : IAllocator
    {
        private readonly GgmlContext context;
        private readonly int deviceId;

        public GgmlAllocator(GgmlContext context, int deviceId)
        {
            this.context = context ?? throw new ArgumentNullException(nameof(context));
            this.deviceId = deviceId;
        }

        public BlasEnum BlasEnum => context.BackendType == GgmlBackendType.Metal ? BlasEnum.GGML_METAL : BlasEnum.GGML_CPU;

        public int DeviceId => deviceId;

        public GgmlContext Context => context;

        public Storage Allocate(DType elementType, long elementCount)
        {
            if (elementType == DType.Float16)
            {
                throw new NotSupportedException("GGML backends currently support Float32 tensors only. Disable AMP to use this backend.");
            }

            return new GgmlStorage(this, context, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
