// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen4ExpArenaBudgetKind")]
    internal static partial int Qwen4ExpArenaBudgetKind();
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginScopedQwen4ExpArena")]
    internal static partial ulong BeginScopedQwen4ExpArena(int maximumEntries);
    [LibraryImport(DllName, EntryPoint = "TSGgml_EndScopedQwen4ExpArena")]
    internal static partial int EndScopedQwen4ExpArena(ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen4ExpDropHolderGraphs")]
    internal static partial int Qwen4ExpDropHolderGraphs(IntPtr attention, IntPtr recurrent, IntPtr ple);
    [LibraryImport(DllName, EntryPoint = "TSGgml_Qwen4ExpArenaLastExecution")]
    internal static partial int Qwen4ExpArenaLastExecution(out ulong executionId, out int sequences,
        out int slots, out int nodes, out int hostSegments, out int path);
}

public readonly record struct GgmlQwen4ExpBatchExecution(ulong ExecutionId, int Sequences,
    int Slots, int Nodes, int HostSegments, int Path);

public partial class GgmlBasicOps
{
    public static void Qwen4ExpDropHolderGraphs(IntPtr attention, IntPtr recurrent, IntPtr ple)
    {
        if (GgmlNative.Qwen4ExpDropHolderGraphs(attention, recurrent, ple) != 1)
            throw new InvalidOperationException("Qwen4Exp holder graph retirement failed: " + GgmlNative.LastNativeError());
    }
    public static GgmlQwen4ExpBatchExecution? TryGetQwen4ExpBatchExecution()
    {
        try
        {
            return GgmlNative.Qwen4ExpArenaLastExecution(out var id, out var sequences, out var slots,
                out var nodes, out var segments, out var path) == 1
                ? new(id, sequences, slots, nodes, segments, path) : null;
        }
        catch (EntryPointNotFoundException) { return null; } // Ordinary legacy deployments retain their ABI.
    }
}

/// <summary>Exclusive retained-arena policy. The native scope flushes dirty
/// holders before retiring arenas; failure preserves its token for retry.
/// Install before model loading and release only after model cleanup. Quotas
/// are supplied separately by GgmlCacheBudgetScope, not by this policy.</summary>
public sealed class GgmlQwen4ExpArenaScope : IDisposable
{
    private ulong _token;
    public GgmlQwen4ExpArenaScope(int maximumEntries = 1)
    {
        if (maximumEntries is < 1 or > 4) throw new ArgumentOutOfRangeException(nameof(maximumEntries));
        try
        {
            if (GgmlNative.Qwen4ExpArenaBudgetKind() != 5)
                throw new NotSupportedException("Unified joint batching requires the dedicated arena allocation kind 5.");
            _token = GgmlNative.BeginScopedQwen4ExpArena(maximumEntries);
            if (_token == 0) throw new InvalidOperationException("Could not acquire the Qwen4Exp joint arena policy: " + GgmlNative.LastNativeError());
        }
        catch (EntryPointNotFoundException error)
        {
            throw new NotSupportedException("Unified joint batching requires matching managed and native arena ownership support.", error);
        }
    }
    public void Dispose()
    {
        if (_token == 0) return;
        if (GgmlNative.EndScopedQwen4ExpArena(_token) != 1)
            throw new InvalidOperationException("The joint arena still owns state; retry cleanup: " + GgmlNative.LastNativeError());
        _token = 0;
    }
}
