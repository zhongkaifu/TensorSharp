// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp
{
    /// <summary>
    /// The memory of the device an allocator places tensors on, for planners that decide what stays resident there (a
    /// MoE model's routed experts) without knowing the backend.
    /// </summary>
    public interface IDeviceMemoryInfo
    {
        /// <summary>The device's free and total bytes; false when the device cannot say. Free counts the cached blocks the
        /// allocator gives back as soon as an allocation needs the room (on CUDA its large-block cache, not the small
        /// blocks its shards keep), so what it reports can be allocated.</summary>
        bool TryGetDeviceMemory(out long freeBytes, out long totalBytes);

        /// <summary>Whether the device's memory is the host's (unified memory): what stays "on the device" then competes
        /// with the host's own pages.</summary>
        bool SharesHostMemory { get; }
    }
}
