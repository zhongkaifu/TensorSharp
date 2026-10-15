// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>The observation used by a GLM admission check, retained so a refusal
/// describes that check rather than a later, potentially different OS sample.
/// This is an observation, not an OS-enforced physical ownership claim.</summary>
internal readonly record struct GlmUnifiedPhysicalObservation(
    InferenceHardwareMemory Hardware, IReadOnlyList<MemoryPoolSnapshot> Ledger,
    MemoryCharge[] Limits, long HostHeadroom, long DeviceHeadroom)
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    internal static GlmUnifiedPhysicalObservation Create(InferenceHardwareMemory hardware,
        IReadOnlyList<MemoryPoolSnapshot> ledger, long hostCeiling, long deviceCeiling,
        long hostHeadroom, long deviceHeadroom)
    {
        ArgumentNullException.ThrowIfNull(ledger);
        ArgumentOutOfRangeException.ThrowIfNegative(hostCeiling);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceCeiling);
        ArgumentOutOfRangeException.ThrowIfNegative(hostHeadroom);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceHeadroom);
        var host = hardware.HostAllocationCapacity();
        if (hardware.DeviceTotal <= 0 || hardware.DeviceAvailable < 0 || hardware.DeviceAvailable > hardware.DeviceTotal)
            throw new ArgumentException("Invalid physical CUDA capacity observation.", nameof(hardware));
        // Committed tracked bytes are already absent from the OS's availability.
        // Unallocated reservation credit is not: adding Reserved here would
        // incorrectly let another request spend that same physical allowance.
        long hostLimit = (long)Math.Min(hostCeiling, (decimal)ledger.Single(p => p.Pool == Host).Committed
            + Math.Max(0, host.Available - hostHeadroom));
        long deviceLimit = (long)Math.Min(deviceCeiling, (decimal)ledger.Single(p => p.Pool == Device).Committed
            + Math.Max(0, hardware.DeviceAvailable - deviceHeadroom));
        return new(hardware, ledger, [new(Host, hostLimit), new(Device, deviceLimit)], hostHeadroom, deviceHeadroom);
    }

    internal bool Fits(IReadOnlyList<MemoryCharge> growth)
    {
        foreach (var charge in growth)
            if (Shortfall(charge.Pool, growth) > 0) return false;
        return true;
    }

    internal decimal Shortfall(string poolName, IReadOnlyList<MemoryCharge> growth)
    {
        var pool = Ledger.Single(p => p.Pool == poolName);
        long limit = Limits.Single(p => p.Pool == poolName).Bytes;
        long delta = growth.Single(c => c.Pool == poolName).Bytes;
        return Math.Max(0, (decimal)pool.Reserved + pool.Committed + delta - Math.Min(pool.Capacity, limit));
    }

    /// <summary>A mapped page hint can improve currently observed RAM, but it
    /// cannot solve a CUDA deficit or grant any owned allocation credit.</summary>
    internal bool TryRefreshMappedHostPressure(IReadOnlyList<MemoryCharge> growth, Func<bool> trimHint,
        Func<GlmUnifiedPhysicalObservation> resample, out GlmUnifiedPhysicalObservation refreshed, out bool? hinted)
    {
        refreshed = this; hinted = null;
        if (Shortfall(Host, growth) <= 0 || Shortfall(Device, growth) > 0) return false;
        hinted = trimHint();
        refreshed = resample();
        return true;
    }

    internal string Describe(IReadOnlyList<MemoryCharge> growth)
    {
        var sample = this;
        string pools = string.Join("; ", growth.Select(charge =>
        {
            var pool = sample.Ledger.Single(p => p.Pool == charge.Pool);
            long limit = sample.Limits.Single(p => p.Pool == charge.Pool).Bytes;
            decimal post = (decimal)pool.Reserved + pool.Committed + charge.Bytes;
            long effective = Math.Min(pool.Capacity, limit);
            return FormattableString.Invariant($"{charge.Pool}: reserved={pool.Reserved}, committed={pool.Committed}, capacity={pool.Capacity}, physical-limit={limit}, growth={charge.Bytes}, combined-post={post}, shortfall={Math.Max(0, post - effective)}");
        }));
        string Commit(long? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "unknown";
        return FormattableString.Invariant($"pools=[{pools}]; physical=[host-total={Hardware.HostTotal}, host-available={Hardware.HostAvailable}, host-commit-limit={Commit(Hardware.HostCommitLimit)}, host-commit-available={Commit(Hardware.HostCommitAvailable)}, host-allocation-available={Hardware.HostAllocationCapacity().Available}, host-headroom={HostHeadroom}, device-total={Hardware.DeviceTotal}, device-available={Hardware.DeviceAvailable}, device-headroom={DeviceHeadroom}]");
    }
}
