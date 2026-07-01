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
            // Float16 storage is permitted only for KV-cache use (the Tensor framework
            // ops still operate on Float32; F16 K/V tensors are read/written via the
            // hand-written conversion helpers in ModelBase + the native flash-attention
            // kernels which understand F16 cache layouts directly).
            //
            // Q8_0 / Q4_0 are even more restricted: only the native GGML kernels can
            // read/write them (they understand the 32-element block layout). Direct
            // element access from C# throws.
            if (elementType != DType.Float32 &&
                elementType != DType.Float16 &&
                elementType != DType.Float64 &&
                elementType != DType.Int32 &&
                elementType != DType.UInt8 &&
                elementType != DType.Q8_0 &&
                elementType != DType.Q4_0)
            {
                throw new NotSupportedException($"GGML backend does not support storage for {elementType}.");
            }

            return new GgmlStorage(this, context, elementType, elementCount);
        }

        public float GetAllocatedMemoryRatio()
        {
            return 0.0f;
        }
    }
}
