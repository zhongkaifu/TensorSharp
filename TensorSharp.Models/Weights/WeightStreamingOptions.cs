// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Opt-in bounded file-weight reads and CUDA linear workspaces. Supported
/// weight formats depend on the model adapter. The shared
/// budget covers the weight staging buffer, optional reusable host payload, and each streamed linear's device input,
/// weight/output staging and arithmetic scratch. Some adapters require a complete
/// temporary device matrix to preserve the resident reduction order. Existing model activations, live KV, small resident
/// parameters, driver/runtime allocations and the OS file cache are not covered.</summary>
public sealed class WeightStreamingOptions
{
    public WeightStreamingOptions(MemoryBudget budget, string hostPool,
        IEnumerable<string> devicePools, int tileBytes = 1 << 20, int tokenTileRows = 32)
        : this(budget, hostPool, devicePools, tileBytes, tokenTileRows, readAhead: true) { }

    public WeightStreamingOptions(MemoryBudget budget, string hostPool,
        IEnumerable<string> devicePools, int tileBytes, int tokenTileRows, bool readAhead)
    {
        Budget = budget ?? throw new ArgumentNullException(nameof(budget));
        ArgumentException.ThrowIfNullOrWhiteSpace(hostPool);
        ArgumentNullException.ThrowIfNull(devicePools);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tileBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(tokenTileRows);
        if (tokenTileRows > 65535 * 8) throw new ArgumentOutOfRangeException(nameof(tokenTileRows));
        string[] pools = devicePools.ToArray();
        if (pools.Length == 0 || pools.Any(string.IsNullOrWhiteSpace) || pools.Distinct(StringComparer.Ordinal).Count() != pools.Length)
            throw new ArgumentException("Specify distinct device capacity constraints for the streaming rank.", nameof(devicePools));
        var known = budget.Snapshot().Select(x => x.Pool).ToHashSet(StringComparer.Ordinal);
        if (!known.Contains(hostPool) || pools.Any(p => !known.Contains(p)))
            throw new ArgumentException("Every streaming pool must exist in the shared budget.");
        HostPool = hostPool;
        DevicePools = Array.AsReadOnly(pools);
        TileBytes = tileBytes;
        TokenTileRows = tokenTileRows;
        ReadAhead = readAhead;
    }

    public MemoryBudget Budget { get; }
    public string HostPool { get; }
    /// <summary>Constraints charged by the same device allocation; do not sum them
    /// as separate copies. The initial model adapter supports one CUDA rank.</summary>
    public IReadOnlyList<string> DevicePools { get; }
    public int TileBytes { get; }
    /// <summary>Maximum tokens in a host transfer or tiled projection. The planner
    /// may reduce it to fit shared capacity. Resident-compatible matrix arithmetic
    /// can additionally require a charged device workspace for the full logical N.</summary>
    public int TokenTileRows { get; }
    /// <summary>When shared RAM permits a second charged tile plus output staging,
    /// start the next file read before computing the current tile. A smaller budget
    /// retains one-buffer execution. This does not imply asynchronous CUDA copies.</summary>
    public bool ReadAhead { get; }

    private long _hostCacheBytes, _hostCacheReserveBytes = 4L << 20;
    /// <summary>Optional ceiling for reusable original weight bytes in pageable RAM.
    /// Actual admission also respects remaining shared capacity and workspace reserve.
    /// Zero preserves uncached execution. Cache entries never own native graph pointers.</summary>
    public long HostCacheBytes
    {
        get => _hostCacheBytes;
        init { ArgumentOutOfRangeException.ThrowIfNegative(value); _hostCacheBytes = value; }
    }
    /// <summary>Shared host capacity left available when admitting optional cache entries.
    /// The adaptive loader supplies its request peak forecast, including untracked state.</summary>
    public long HostCacheReserveBytes
    {
        get => _hostCacheReserveBytes;
        init { ArgumentOutOfRangeException.ThrowIfNegative(value); _hostCacheReserveBytes = value; }
    }
}

/// <summary>Payload and I/O counters, not process RSS or total CUDA consumption.</summary>
public readonly record struct WeightStreamingStatistics(long FileBackedWeightBytes,
    long FileBytesRead, long LinearTiles, long EmbeddingRows, long PeakHostStagingBytes,
    long PeakDeviceWorkspaceBytes, long DeviceSessionCreations = 0, long InputUploads = 0,
    long CompleteMatrixProjections = 0)
{
    public long ReadAheadOperations { get; init; }
    public long HostCacheBytes { get; init; }
    public long PeakHostCacheBytes { get; init; }
    public long HostCacheHitBytes { get; init; }
    public long HostCacheHits { get; init; }
    public long HostCacheEvictedBytes { get; init; }
}
