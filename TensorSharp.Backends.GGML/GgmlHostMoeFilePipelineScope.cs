// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFilePipelineVersion")]
    internal static partial int HostMoeFilePipelineVersion();
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeFilePipelineScope")]
    internal static partial int BeginHostMoeFilePipelineScope(int enabled, int minimumRows,
        long minimumPayloadBytes, long maximumRingBytes, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_EndHostMoeFilePipelineScope")]
    internal static partial int EndHostMoeFilePipelineScope(ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeFilePipelineStats")]
    internal static partial int HostMoeFilePipelineStats(out long calls, out long sourceBytes, out long retainedRingBytes);
}

/// <summary>Successful file-pipeline device executions and logical source bytes,
/// cumulative for this process. RetainedRingBytes is the current charged pinned
/// buffer payload. These counters prove path selection, not measured overlap,
/// whole-model memory ownership or performance improvement.</summary>
public readonly record struct GgmlHostMoeFilePipelineUsage(long Calls, long SourceBytes, long RetainedRingBytes);

public partial class GgmlBasicOps
{
    public static GgmlHostMoeFilePipelineUsage GetHostMoeFilePipelineUsage()
    {
        if (GgmlNative.HostMoeFilePipelineStats(out long calls, out long sourceBytes, out long ringBytes) != 1)
            throw new InvalidOperationException("GGML file pipeline usage read failed: " + GgmlNative.LastNativeError());
        return new(calls, sourceBytes, ringBytes);
    }
}

internal interface IHostMoeFilePipelineScopeNative
{
    int ParallelReaderVersion() => 0;
    ulong Begin(bool enabled, int minimumRows, long minimumPayloadBytes, long maximumRingBytes);
    void End(ulong token);
}

/// <summary>Exclusive session configuration for bounded asynchronous file-backed
/// expert prefill. It overrides the legacy process environment while owned and
/// allocates no ring at construction. Only admitted single-device CUDA file
/// graphs meeting both thresholds can activate it; decode, mmap, CPU and TP
/// paths stay unchanged. The two pinned slots are separately charged to the
/// shared host budget and together cannot exceed MaximumRingBytes. Vendor pools
/// and OS page residency remain outside this payload ceiling. Install the shared
/// budget first, quiesce all work before Begin/Dispose, and dispose after model
/// cleanup while that budget is still attached. Failed disposal retains its
/// token for retry; no finalizer performs GPU work.</summary>
public sealed class GgmlHostMoeFilePipelineScope : IDisposable
{
    private readonly object _gate = new();
    private readonly IHostMoeFilePipelineScopeNative _native;
    private readonly ulong _token;
    private bool _disposed;

    public GgmlHostMoeFilePipelineScope(bool enabled, int minimumRows = 64,
        long minimumPayloadBytes = 32L << 20, long maximumRingBytes = 16L << 20)
        : this(enabled, minimumRows, minimumPayloadBytes, maximumRingBytes, NativeApi.Instance) { }

    internal GgmlHostMoeFilePipelineScope(bool enabled, int minimumRows,
        long minimumPayloadBytes, long maximumRingBytes, IHostMoeFilePipelineScopeNative native)
    {
        if (minimumRows is <= 32 or > 1048576) throw new ArgumentOutOfRangeException(nameof(minimumRows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(minimumPayloadBytes);
        if (maximumRingBytes < 128 || maximumRingBytes > (16L << 20) || maximumRingBytes % 128 != 0)
            throw new ArgumentOutOfRangeException(nameof(maximumRingBytes));
        ArgumentNullException.ThrowIfNull(native);
        if (enabled && native.ParallelReaderVersion() < 2)
            throw new NotSupportedException("Asynchronous expert prefill requires the qualified parallel-reader native capability.");
        _native = native;
        Enabled = enabled; MinimumRows = minimumRows;
        MinimumPayloadBytes = minimumPayloadBytes; MaximumRingBytes = maximumRingBytes;
        // Complete managed setup before the exclusive native ownership changes.
        _token = native.Begin(enabled, minimumRows, minimumPayloadBytes, maximumRingBytes);
    }
    public bool Enabled { get; }
    public int MinimumRows { get; }
    public long MinimumPayloadBytes { get; }
    public long MaximumRingBytes { get; }
    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _native.End(_token);
            _disposed = true;
        }
    }
    private sealed class NativeApi : IHostMoeFilePipelineScopeNative
    {
        public static readonly NativeApi Instance = new();
        private static void Require(int status)
        {
            if (status != 1) throw new InvalidOperationException("GGML file pipeline scope failed: " + GgmlNative.LastNativeError());
        }
        public int ParallelReaderVersion()
        {
            try { return GgmlNative.HostMoeFilePipelineVersion(); }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Parallel expert prefill requires the matching updated GgmlOps native library.", error);
            }
        }
        public ulong Begin(bool enabled, int minimumRows, long minimumPayloadBytes, long maximumRingBytes)
        {
            try
            {
                Require(GgmlNative.BeginHostMoeFilePipelineScope(enabled ? 1 : 0, minimumRows,
                    minimumPayloadBytes, maximumRingBytes, out ulong token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("File pipeline scopes require the matching updated GgmlOps native library.", error);
            }
        }
        public void End(ulong token) => Require(GgmlNative.EndHostMoeFilePipelineScope(token));
    }
}
