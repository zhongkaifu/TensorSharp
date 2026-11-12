// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Models;

/// <summary>
/// The on-node group of a cross-node layer split on <c>--backend cuda</c>: only the GPU count. The direct-CUDA
/// engines create their own per-device contexts, allocators and streams, so this group initializes no ggml
/// backend, no meta-backend and no peer access, and it never reduces anything: whole-layer stages exchange their
/// boundary state through the node pipeline (<see cref="ILayerSplitGroup.Send"/> / Receive).
/// </summary>
public sealed class DirectCudaLayerSplitDevices : ITensorParallelGroup
{
    public DirectCudaLayerSplitDevices(int degree)
    {
        if (degree is < 1 or > 8)
            throw new ArgumentOutOfRangeException(nameof(degree), "--layer-split takes 1..8 GPUs per node.");
        int visible;
        try { visible = CudaDevice.GetDeviceCount(); }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or CudaException or TypeInitializationException)
        {
            throw new NotSupportedException("--backend cuda needs a CUDA driver, and none could be initialized: " + ex.Message, ex);
        }
        if (visible < degree)
            throw new NotSupportedException($"--layer-split {degree} asks for {degree} CUDA device(s) on this node, but {visible} are visible.");
        Degree = degree;
    }

    public int Degree { get; }
    public int GlobalDegree => Degree;
    public int GlobalRankOffset => 0;
    public int NodeCount => 1;
    // Placement only: never activates tensor sharding.
    public bool IsActive => false;

    private readonly IAllocator[] _allocators = new IAllocator[8];

    /// <summary>Local device <paramref name="rank"/>'s allocator (TENSORSHARP_LAYER_SPLIT_DEVICES picks the ordinals, as
    /// for a local split), made on first request: an engine that owns its device contexts never asks, the shared decoder
    /// runtime's stages do.</summary>
    public IAllocator GetAllocator(int rank)
    {
        if ((uint)rank >= (uint)Degree) throw new ArgumentOutOfRangeException(nameof(rank));
        lock (_allocators)
            return _allocators[rank] ??= new CudaAllocator(ModelBase.ParseDeviceIds(Degree, "TENSORSHARP_LAYER_SPLIT_DEVICES")[rank]);
    }

    public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Layer splitting does not reduce tensors.");
    public void Synchronize() { }
    public void Barrier() { }
    public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
    public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
    public void Dispose()
    {
        lock (_allocators)
            for (int i = 0; i < _allocators.Length; i++)
            {
                (_allocators[i] as IDisposable)?.Dispose();
                _allocators[i] = null;
            }
    }
}
