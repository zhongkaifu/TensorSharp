// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;

namespace TensorSharp
{
    /// <summary>
    /// An allocator whose device copies from host memory run at the link's DMA speed only from page-locked memory (a
    /// discrete GPU staging a model's offloaded experts from the model file's mapping). A planner hands it the ranges it
    /// will stage from, once, off the request path.
    /// </summary>
    public interface IHostRangePinner
    {
        /// <summary>Page-lock as many of <paramref name="ranges"/> as the host allows (the rest stay pageable), in order,
        /// stopping at <paramref name="cancel"/>. <paramref name="allowance"/>: the bytes the caller measured free for the
        /// locks (0: the pinner's own limit), which the process may lock from then on. The handle unlocks them: dispose it
        /// before the memory goes. Any thread.</summary>
        IDisposable PinHostRanges(IReadOnlyList<(IntPtr Pointer, long Bytes)> ranges, long allowance, CancellationToken cancel);
    }
}
