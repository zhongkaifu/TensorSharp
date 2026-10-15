// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Paged;

/// <summary>Bounds the executor's host KV snapshots, capture scratch and spill staging.
/// Does not include the model's live device KV, native holders, weights or OS page cache.</summary>
/// <remarks>When configured on an engine, selects the per-sequence snapshot/swap route.
/// Model-owned paged arrays and per-request fused holders cannot use this storage.</remarks>
public sealed record KvSnapshotOptions(long RamBytes, long SsdBytes, string SpillDirectory,
    int TransferBytes = 1 << 20)
{
    /// <summary>Borrowed process-wide budget when created with FromSharedBudget.
    /// Its named pools replace the legacy per-engine byte limits. RamBytes and
    /// SsdBytes then describe pool capacities at configuration time, not subquotas.
    /// Disposing the engine releases only its own allocations.</summary>
    public MemoryBudget? SharedBudget { get; private init; }
    public string RamPool { get; private init; } = "kv/ram";
    public string SsdPool { get; private init; } = "kv/ssd";

    /// <summary>Charge snapshot pages, capture scratch, transfer buffers and spill
    /// files to existing physical pools, also usable by request/native budgets.
    /// No whole-pool reservation or per-engine subquota is created. An engine can
    /// evict its own snapshots; it cannot reclaim another owner's resources.</summary>
    public static KvSnapshotOptions FromSharedBudget(MemoryBudget budget, string ramPool,
        string ssdPool, string spillDirectory, int transferBytes = 1 << 20)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentException.ThrowIfNullOrWhiteSpace(ramPool);
        ArgumentException.ThrowIfNullOrWhiteSpace(ssdPool);
        if (StringComparer.Ordinal.Equals(ramPool, ssdPool))
            throw new ArgumentException("RAM and spill storage must use distinct physical pools.");
        var pools = budget.Snapshot();
        long Capacity(string pool)
        {
            foreach (var entry in pools)
                if (entry.Pool == pool) return entry.Capacity;
            throw new ArgumentException($"Unknown snapshot memory pool: {pool}");
        }
        return new(Capacity(ramPool), Capacity(ssdPool), spillDirectory, transferBytes)
        {
            SharedBudget = budget, RamPool = ramPool, SsdPool = ssdPool,
        };
    }

    internal void Validate(long blockBytes)
    {
        // Re-read shared capacities: configuration can predate a capacity change.
        var pools = SharedBudget?.Snapshot();
        long ramBytes = pools?.Single(p => p.Pool == RamPool).Capacity ?? RamBytes;
        long ssdBytes = pools?.Single(p => p.Pool == SsdPool).Capacity ?? SsdBytes;
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(ramBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(ssdBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(TransferBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(SpillDirectory);
        if (blockBytes > int.MaxValue)
            throw new NotSupportedException("The model snapshot contract uses Span<byte>; a KV block must fit in Int32. Reduce the block size.");
        long alignedBlock = checked((blockBytes + 63) / 64 * 64);
        long minimum = checked(2 * alignedBlock + ((long)TransferBytes + 63) / 64 * 64);
        if (ramBytes < minimum)
            throw new ArgumentException($"KV snapshot RAM requires at least {minimum} bytes for one page, capture scratch and staging.");
    }

    /// <summary>Explicit, fail-fast byte budgets. Unset RAM means the legacy storage path.</summary>
    public static KvSnapshotOptions? FromEnvironment()
    {
        string? ram = Environment.GetEnvironmentVariable("TS_SCHED_KV_RAM_BYTES");
        string? ssd = Environment.GetEnvironmentVariable("TS_SCHED_KV_SSD_BYTES");
        string? directory = Environment.GetEnvironmentVariable("TS_SCHED_KV_SPILL_DIRECTORY");
        if (string.IsNullOrWhiteSpace(ram))
        {
            if (!string.IsNullOrWhiteSpace(ssd) || !string.IsNullOrWhiteSpace(directory))
                throw new ArgumentException("Set TS_SCHED_KV_RAM_BYTES to enable bounded KV snapshots.");
            return null;
        }
        static long Parse(string? value, string name, bool zeroAllowed = false)
            => long.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out long bytes) && (bytes > 0 || (zeroAllowed && bytes == 0))
                ? bytes : throw new ArgumentException($"{name} must be a positive byte count.");
        return new(Parse(ram, "TS_SCHED_KV_RAM_BYTES"),
            string.IsNullOrWhiteSpace(ssd) ? 0 : Parse(ssd, "TS_SCHED_KV_SSD_BYTES", zeroAllowed: true),
            string.IsNullOrWhiteSpace(directory) ? Path.Combine(Path.GetTempPath(), "tensorsharp-kv") : directory);
    }
}
