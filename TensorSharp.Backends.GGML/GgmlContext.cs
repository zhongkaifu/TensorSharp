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
using System.Reflection;

namespace TensorSharp.GGML
{
    public sealed class GgmlContext
    {
        internal GgmlMemoryPool MemoryPool { get; }

        public GgmlContext(int[] deviceIds, GgmlBackendType backendType)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            if (deviceIds.Length != 1)
            {
                throw new NotSupportedException("GGML backends currently support a single device only.");
            }

            DeviceId = deviceIds[0];
            BackendType = backendType;
            MemoryPool = new GgmlMemoryPool(backendType);
            MemoryPool.EnsureInitialBlocks();
            GgmlNative.EnsureAvailable(backendType);
            OpRegistry.RegisterAssembly(Assembly.GetExecutingAssembly());
        }

        public int DeviceId { get; }

        public GgmlBackendType BackendType { get; }
    }
}
