// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Cuda
{
    /// <summary>
    /// Holds device memory it can give back when an allocation on a <see cref="CudaAllocator"/> finds the device full (a
    /// model's retained conversations, say). Registered with <see cref="CudaAllocator.RegisterReclaimer"/>, it is asked
    /// after the allocator's cached blocks went back to the driver and before the allocator's last resort (a full GC and a
    /// drain of every cached block); never while a CUDA graph is being captured on the allocator, and never from inside
    /// another reclaimer's call on the same thread.
    /// </summary>
    public interface IDeviceMemoryReclaimer
    {
        /// <summary>
        /// Release device memory towards <paramref name="bytes"/> (the size of the allocation that failed), giving up only
        /// what no work in flight uses. Runs on the allocating thread, which may be any thread: a reclaimer that may act only
        /// on certain threads or outside its own critical sections decides that itself and returns 0. Returns the bytes it
        /// released (they may be back in the allocator's cache, which the allocator then trims).
        /// </summary>
        long Reclaim(long bytes);
    }
}
