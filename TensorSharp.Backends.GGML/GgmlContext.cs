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

        /// <summary>
        /// Return every pooled-but-unused host block to the operating system and report
        /// how many bytes that was. The pool keeps freed blocks so the next allocation
        /// is cheap; on a device that is about to be killed for memory, cheap is not
        /// the point.
        /// </summary>
        public long ReleasePooledMemory() => MemoryPool.Trim();

        /// <summary>
        /// The model that owns this context has been disposed: release the pooled host
        /// blocks and stop pooling (see <see cref="GgmlMemoryPool.Close"/>). Only the owner
        /// may call this, after its tensors are freed and no Metal buffer wraps them any
        /// longer (ClearHostBufferCache).
        /// </summary>
        public long ReleasePoolForDisposal() => MemoryPool.Close();

        public GgmlContext(int[] deviceIds, GgmlBackendType backendType)
            : this(deviceIds, backendType, enableCollectives: true)
        {
        }

        /// <param name="enableCollectives">
        /// False for a LAYER SPLIT: bring up one backend per GPU but create no
        /// cross-device collective. Each GPU runs a contiguous run of layers and
        /// the only thing that crosses a boundary is the residual, handed over
        /// through host memory, so there is nothing to AllReduce.
        /// </param>
        public GgmlContext(int[] deviceIds, GgmlBackendType backendType, bool enableCollectives)
        {
            if (deviceIds == null || deviceIds.Length == 0)
            {
                throw new ArgumentException("At least one device id is required for the GGML backend.", nameof(deviceIds));
            }

            DeviceIds = (int[])deviceIds.Clone();
            DeviceId = deviceIds[0];
            BackendType = backendType;
            MemoryPool = new GgmlMemoryPool(backendType);
            GgmlNative.EnsureAvailable(backendType);

            if (deviceIds.Length > 1)
            {
                // Several GPUs: bring up one ggml backend per GPU. Ops then select a
                // rank with GgmlNative.SetActiveDevice; tensors carry their rank
                // through GgmlAllocator.DeviceId. Used both by tensor parallelism
                // (every GPU holds a shard of every weight, collectives on) and by a
                // layer split (each GPU holds a run of whole layers, collectives off).
                if (backendType != GgmlBackendType.Cuda && backendType != GgmlBackendType.Vulkan)
                {
                    throw new NotSupportedException(
                        $"The GGML {backendType} backend exposes a single device; multi-GPU requires the CUDA or Vulkan backend.");
                }
                // The native side needs to know whether the ranks will be driven
                // concurrently: that decides whether ggml-cuda's graph capture
                // (which is process-wide and breaks under concurrent CUDA calls)
                // has to be turned off for the run.
                if (enableCollectives)
                {
                    GgmlNative.TensorParallelInit(backendType, DeviceIds, GgmlTensorParallelGroup.ParallelRanks);
                    HasDeviceAllReduce = GgmlNative.TensorParallelHasDeviceAllReduce();
                }
                else
                {
                    GgmlNative.MultiDeviceInit(backendType, DeviceIds);
                    HasDeviceAllReduce = false;
                }
            }
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
            // Set TS_GGML_ASYNC_COMPUTE=0 to disable and fall back to the
            // eager-sync behaviour for debugging.
            var disableAsync = Environment.GetEnvironmentVariable("TS_GGML_ASYNC_COMPUTE");
            bool enableAsync = backendType == GgmlBackendType.Metal &&
                               !string.Equals(disableAsync, "0", StringComparison.Ordinal);
            GgmlNative.SetAsyncCompute(enableAsync);
            // Allocate initial host blocks only after fallible backend setup.
            MemoryPool.EnsureInitialBlocks();
        }

        /// <summary>Physical device index backing rank 0.</summary>
        public int DeviceId { get; }

        /// <summary>Physical device index per rank; length == the TP degree.</summary>
        public int[] DeviceIds { get; }

        /// <summary>Number of GPUs this context spans (1 = no tensor parallelism).</summary>
        public int Degree => DeviceIds.Length;

        /// <summary>
        /// True when the backend collective passes its execution probe. This
        /// includes NCCL, P2P and ggml's pinned-host pipeline; it does not prove
        /// that transport stays on the GPUs. Native diagnostics report the route.
        /// </summary>
        public bool HasDeviceAllReduce { get; }

        public GgmlBackendType BackendType { get; }
    }
}
