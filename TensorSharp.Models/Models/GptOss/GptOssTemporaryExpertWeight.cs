// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.GGML;

namespace TensorSharp.Models;

/// <summary>One bounded expert gate||up copy. Retires native cache references
/// before freeing the host allocation, including after an execution exception.</summary>
internal sealed class GptOssTemporaryExpertWeight : IDisposable
{
    private readonly Action _barrier;
    private readonly Action<IntPtr> _invalidate;
    internal QuantizedWeight Weight { get; private set; }

    internal unsafe GptOssTemporaryExpertWeight(QuantizedWeight gate, QuantizedWeight up, bool ggml,
        Action barrier = null, Action<IntPtr> invalidate = null)
    {
        if (gate == null || up == null || gate.GgmlType != up.GgmlType ||
            gate.Ne0 <= 0 || gate.Ne0 != up.Ne0 || gate.Ne1 <= 0 || gate.Ne1 != up.Ne1 ||
            gate.Ne1 > int.MaxValue / 2 || gate.Scale != 1f || up.Scale != 1f ||
            gate.RawBytes <= 0 || gate.RawBytes != up.RawBytes ||
            !gate.HasHostData || !up.HasHostData)
            throw new ArgumentException("Temporary expert fusion requires compatible, unit-scale host weights.");
        _barrier = barrier ?? (ggml ? GgmlBasicOps.HostReadBarrier : static () => { });
        _invalidate = invalidate ?? (ggml ? GgmlBasicOps.InvalidateHostBuffer : static _ => { });
        // Always own a distinct pointer. Even adjacent source views cannot share
        // their cache identity with a temporary tensor of a different shape.
        long bytes = checked(gate.RawBytes + up.RawBytes);
        IntPtr allocation = QuantizedWeight.AllocateBuffer(bytes);
        try
        {
            Buffer.MemoryCopy((void*)gate.Data, (void*)allocation, bytes, gate.RawBytes);
            Buffer.MemoryCopy((void*)up.Data, (byte*)allocation + gate.RawBytes, up.RawBytes, up.RawBytes);
            Weight = new QuantizedWeight(allocation, bytes, gate.GgmlType, gate.Ne0, checked(gate.Ne1 * 2));
            allocation = IntPtr.Zero;
        }
        finally { if (allocation != IntPtr.Zero) QuantizedWeight.FreeBuffer(allocation); }
    }

    public void Dispose()
    {
        if (Weight == null) return;
        // CacheKey may differ from Data after a preload. Never return this host
        // address to the allocator while a native cache entry can still use it.
        _barrier();
        _invalidate(Weight.CacheKey);
        Weight.Dispose();
        Weight = null;
    }
}
