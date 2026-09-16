// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Optional, non-mutating admission checks for a model that can restore a
    /// particular complete continuation holder beyond the scheduler's short
    /// control-token rewind limit. Checks must identify the actual source,
    /// reject failed/head-mismatched state, and never select or mutate it.
    /// Execution still rechecks the restore after ownership is transferred.
    /// </summary>
    public interface IExactFusedCacheReuse
    {
        bool SupportsExactFusedCacheReuse { get; }
        bool CanReuseLivePrefix(int cachedTokenCount, int targetTokenCount);
        bool CanReuseRetainedPrefix(string retainedKey, int cachedTokenCount, int targetTokenCount);
    }
}
