// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.Models;

/// <summary>Conservative starting placement for public copied checkpoints,
/// using geometry and an effective device ceiling rather than a model name.
/// The caller must still fit mandatory execution floors and account each live
/// owner. The one-eighth count allowance is not a physical hard limit or a
/// promise that an optional replacement fits alongside live execution.</summary>
internal readonly record struct UnifiedMemoryPrefixCapacity(int RetainedCopies, long PlacementReserveBytes)
{
    internal static UnifiedMemoryPrefixCapacity Select(int requested, long bytesPerCopy, long deviceCeiling)
    {
        if (requested < 0 || requested > 8)
            throw new ArgumentOutOfRangeException(nameof(requested));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bytesPerCopy);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceCeiling);

        long allowance = deviceCeiling / 8;
        // Bound the retained count with one possible replacement before
        // multiplying. Placement protects retained owners only: making an
        // optional replacement part of the immutable execution floor can
        // unnecessarily reduce prefill width for the model's entire lifetime.
        // Every replacement still needs its own fresh physical admission;
        // future reclamation never grants present credit.
        int capacity = (int)Math.Min((long)requested + 1, allowance / bytesPerCopy);
        int retained = Math.Max(0, capacity - 1);
        long reserve = retained * bytesPerCopy;
        return new(retained, reserve);
    }
}
