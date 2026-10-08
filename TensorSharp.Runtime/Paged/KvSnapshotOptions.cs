// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Globalization;
using System.IO;

namespace TensorSharp.Runtime.Paged;

/// <summary>Bounds the executor's host KV snapshots, capture scratch and spill staging.
/// Does not include the model's live device KV, native holders, weights or OS page cache.</summary>
/// <remarks>When configured on an engine, selects the per-sequence snapshot/swap route.
/// Model-owned paged arrays and per-request fused holders cannot use this storage.</remarks>
public sealed record KvSnapshotOptions(long RamBytes, long SsdBytes, string SpillDirectory,
    int TransferBytes = 1 << 20)
{
    internal void Validate(long blockBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(RamBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(SsdBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(TransferBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(SpillDirectory);
        if (blockBytes > int.MaxValue)
            throw new NotSupportedException("The model snapshot contract uses Span<byte>; a KV block must fit in Int32. Reduce the block size.");
        long alignedBlock = checked((blockBytes + 63) / 64 * 64);
        long minimum = checked(2 * alignedBlock + ((long)TransferBytes + 63) / 64 * 64);
        if (RamBytes < minimum)
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
