// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Opt-in bounded Q8_0 weight reads and CUDA linear workspaces. The shared
/// budget covers the weight staging buffer and each streamed linear's device input,
/// weight tile and output tile. Existing model activations, live KV, small resident
/// parameters, driver/runtime allocations and the OS file cache are not covered.</summary>
public sealed class WeightStreamingOptions
{
    public WeightStreamingOptions(MemoryBudget budget, string hostPool,
        IEnumerable<string> devicePools, int tileBytes = 1 << 20, int tokenTileRows = 32)
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
    }

    public MemoryBudget Budget { get; }
    public string HostPool { get; }
    /// <summary>Constraints charged by the same device allocation; do not sum them
    /// as separate copies. The initial model adapter supports one CUDA rank.</summary>
    public IReadOnlyList<string> DevicePools { get; }
    public int TileBytes { get; }
    /// <summary>Maximum tokens sharing a weight tile. The planner reduces this
    /// when the remaining shared capacity cannot hold the input/output workspace.</summary>
    public int TokenTileRows { get; }
}

/// <summary>Payload and I/O counters, not process RSS or total CUDA consumption.</summary>
public readonly record struct WeightStreamingStatistics(long FileBackedWeightBytes,
    long FileBytesRead, long LinearTiles, long EmbeddingRows, long PeakHostStagingBytes,
    long PeakDeviceWorkspaceBytes);
