// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.MLX;

/// <summary>Reads allocator counters without evaluating lazy arrays or waiting
/// for Metal completion. Failure invalidates the entire sample, including any
/// counters already read successfully. Never infer zero from a native error.</summary>
internal static class MlxMemorySnapshotReader
{
    internal delegate int Counter(ref nuint value);
    internal readonly record struct Result(bool Success, MlxMemorySnapshot Snapshot, string Error);

    internal static Result Read(Counter active, Counter cache, Counter peak, Func<string> takeError)
    {
        try
        {
            nuint activeBytes = 0, cacheBytes = 0, peakBytes = 0;
            int status = active(ref activeBytes);
            if (status != 0) return Failed("active", status, takeError());
            status = cache(ref cacheBytes);
            if (status != 0) return Failed("cache", status, takeError());
            status = peak(ref peakBytes);
            if (status != 0) return Failed("peak", status, takeError());
            return new(true, new MlxMemorySnapshot(activeBytes, cacheBytes, peakBytes), null);
        }
        catch (Exception error) when (IsObservationError(error))
        {
            return Unavailable(error);
        }
    }

    private static Result Failed(string counter, int status, string detail) => new(false, default,
        $"MLX {counter} memory is unknown (native status {status})" +
        (string.IsNullOrWhiteSpace(detail) ? "." : $": {detail}"));

    internal static bool IsObservationError(Exception error) =>
        error is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException
            or InvalidOperationException or NotSupportedException;

    internal static Result Unavailable(Exception error) => new(false, default,
        $"MLX allocator memory is unknown: {error.Message}");
}
