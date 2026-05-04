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

            // On Metal, default to async (lazy) GPU dispatch. This is the same model
            // llama.cpp uses: per-op `ggml_metal_graph_compute` commits its command
            // buffer and returns immediately; only the final `ggml_backend_synchronize`
            // (or a host-side data read via TensorComputePrimitives.GetFloatPointer)
            // actually waits on the GPU. With this enabled we avoid one
            // `[cmd_buf waitUntilCompleted]` (~30-100µs of driver/IPC round-trip on
            // M-series Macs) per op submitted, which dominates prefill on long
            // prompts where TensorSharp's per-op driving model would otherwise
            // submit hundreds of command buffers serially.
            //
            // Set TS_GGML_ASYNC_COMPUTE=0 to disable and fall back to the legacy
            // eager-sync behaviour for debugging.
            var disableAsync = Environment.GetEnvironmentVariable("TS_GGML_ASYNC_COMPUTE");
            bool enableAsync = backendType == GgmlBackendType.Metal &&
                               !string.Equals(disableAsync, "0", StringComparison.Ordinal);
            GgmlNative.SetAsyncCompute(enableAsync);
        }

        public int DeviceId { get; }

        public GgmlBackendType BackendType { get; }
    }
}
