// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// A decode step of the shared runtime recorded as one CUDA graph (<see cref="IDecodeCapture"/>). The step's kernels are
    /// launched one by one on the host otherwise - about a thousand a token on Qwen3.5-0.8B, which bound its decode to the
    /// host. The recording owns the pool blocks its kernels use (<see cref="CudaPrefillGraphCache"/>); each sequence's
    /// attention length and cache write row come from its blocks of a dynamic-parameter block the graph uploads first
    /// (<see cref="CudaDecodeDynParams"/>), which the decode attention and the cache append read on the device, naming the
    /// sequence by the cache they read or write.
    /// </summary>
    internal sealed class RuntimeDecodeCapture : IDecodeCapture
    {
        private const string Key = "runtime-decode";

        private readonly CudaAllocator _allocator;
        private readonly CudaPrefillGraphCache _graphs;
        private readonly CudaDecodeDynParams _dyn;
        private readonly Dictionary<int, Tensor> _stable = new();
        private readonly HashSet<int> _varying = new();
        private bool _disposed;

        /// <summary>The varying inputs' bytes in one pinned buffer (offset per slot): the recording copies each from there
        /// (a copy node a replay runs on the device), so a replay makes no upload calls - 54 a token on Qwen3.8 Flash Next
        /// had cost ~0.5 ms of host time between its steps.</summary>
        private IntPtr _pinned;
        private readonly Dictionary<int, long> _pinnedAt = new();

        public RuntimeDecodeCapture(CudaAllocator allocator)
        {
            _allocator = allocator;
            _graphs = new CudaPrefillGraphCache(allocator) { KeepTemplates = true };
            _dyn = new CudaDecodeDynParams(allocator);
        }

        /// <summary>The state leaves the recording was made over: (device base, bytes, byte offset of the leaf) each.</summary>
        private (long Base, long Bytes, long Offset)[] _states;
        private CudaGraphRelocation _relocation;

        public void NoteStates(IReadOnlyList<Tensor> states)
        {
            if (Captured) return;
            _states = Describe(states);
        }

        /// <summary>Each leaf's storage range and offset; null when one is not a device tensor of this allocator.</summary>
        private (long, long, long)[] Describe(IReadOnlyList<Tensor> states)
        {
            var described = new (long, long, long)[states.Count];
            for (int i = 0; i < states.Count; i++)
            {
                if (states[i]?.Storage is not CudaStorage storage || !ReferenceEquals(storage.Allocator, _allocator)) return null;
                described[i] = (storage.DeviceBuffer.ToInt64(), storage.ByteLength, states[i].StorageOffset * states[i].ElementType.Size());
            }
            return described;
        }

        public bool TryRelocate(IReadOnlyList<Tensor> states, IReadOnlyList<Tensor> written)
        {
            if (!Captured || _disposed || _states == null || !_graphs.TryGetGraphs(Key, out IntPtr exec, out IntPtr template)
                || template == IntPtr.Zero)
                return false;
            (long Base, long Bytes, long Offset)[] target = Describe(states);
            if (target == null || target.Length != _states.Length) return false;
            var bases = new long[target.Length];
            var moved = new Dictionary<long, long>();
            for (int i = 0; i < target.Length; i++)
            {
                if (target[i].Bytes != _states[i].Bytes || target[i].Offset != _states[i].Offset) return false;
                // Leaves that shared a storage share one where they move to.
                if (moved.TryGetValue(_states[i].Base, out long to) ? to != target[i].Base : !moved.TryAdd(_states[i].Base, target[i].Base))
                    return false;
                bases[i] = target[i].Base;
            }
            _allocator.Context.MakeCurrent();
            _relocation ??= new CudaGraphRelocation(template, Array.ConvertAll(_states, s => (s.Base, s.Bytes)));
            if (_relocation.Refusal != null)
            {
                if (!_refusalReported)
                {
                    _refusalReported = true;
                    if (RelocationLog) Console.Error.WriteLine($"[cuda-graph] a recorded step cannot move to another slot: {_relocation.Refusal}");
                }
                return false;
            }
            // What the step reads must be on the device (the eager run it replaces would have uploaded it).
            foreach (Tensor t in states) ((CudaStorage)t.Storage).EnsureDeviceCurrent();
            if (!_relocation.Apply(exec, _allocator.Context.Handle, bases))
            {
                // Part of the graph points at the new slot: it must never run again.
                Captured = false;
                return false;
            }
            _graphs.ReplaceKeepAlive(Key, written);
            return true;
        }

        private bool _refusalReported;

        /// <summary>Free device memory below which a recording first returns the pool's cached blocks to the driver.</summary>
        private const long CaptureFreeFloorBytes = 768L << 20;

        /// <summary>Cached large bytes past which a recording checks the driver's free memory first.</summary>
        private const long CaptureTrimCachedBytes = 256L << 20;

        /// <summary>
        /// A recording allocates its step's intermediates while it captures, when the pool can no longer free what it caches:
        /// a prompt's chunk activations stay cached in sizes a decode step never asks for. After a 4K prompt on an A40 holding
        /// 38 GB of Qwen3.8 Flash Next's experts the driver had 97 MiB free beside 1.5 GB cached, the decode steps' recordings
        /// ran out of memory and every step ran unrecorded (34 tok/s against 59). Cached large blocks go back first, the
        /// least recently used first and only until the floor is free: a full drain here freed every cached block, small
        /// ones the decode step rents again included, one synchronizing cuMemFree each, before every new recording on a
        /// nearly full device.
        /// Only a prompt's worth of cached blocks is worth asking the driver about (cuMemGetInfo took ~3 ms on an RTX PRO
        /// 4000; a recording is made at every new width).
        /// </summary>
        private void ReleaseCachedBlocksWhenShort()
        {
            if (_allocator.ReclaimableBytes < CaptureTrimCachedBytes) return;
            long free = _allocator.GetMemoryInfo().freeBytes;
            if (free >= CaptureFreeFloorBytes) return;
            _allocator.TrimLargeCache(CaptureFreeFloorBytes - free);
        }

        private static readonly bool RelocationLog = Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_LOG") == "1";

        public bool Captured { get; private set; }

        public Tensor Stable(int slot, DType type, long[] sizes, bool varies = false)
        {
            if (varies) _varying.Add(slot);
            if (_stable.TryGetValue(slot, out Tensor t))
            {
                if (t.ElementType == type && t.Sizes.SequenceEqual(sizes)) return t;
                if (Captured) throw new InvalidOperationException("a recorded decode step's input changed its shape");
                t.Dispose();
            }
            t = new Tensor(_allocator, type, sizes);
            _stable[slot] = t;
            return t;
        }

        public bool CaptureAndLaunch(Func<bool> run, ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows,
            IReadOnlyList<IReadOnlyList<Tensor>> sequences, IReadOnlyList<Tensor> written)
        {
            if (Captured || _disposed || !_graphs.IsUsable || !_dyn.IsValid || _stable.Count == 0 || attendLengths.Length < 1
                || attendLengths.Length > CudaDecodeDynParams.MaxSequences || writeRows.Length != attendLengths.Length
                || sequences == null || sequences.Count != attendLengths.Length)
                return false;
            _allocator.OnRecordedStepLaunching();
            ReleaseCachedBlocksWhenShort();
            PinVaryingInputs();
            Upload();
            _dyn.Write(attendLengths, writeRows);
            _dyn.NameSequences(sequences);
            if (!_graphs.BeginCapture(Key)) return false;
            bool ok;
            try
            {
                _dyn.EnqueueUpload();
                EnqueueVaryingCopies();
                _dyn.Activate();
                ok = run();
            }
            catch (CudaGraphCaptureAbortedException aborted)
            {
                CudaDecodeDynParams.Deactivate();
                _graphs.AbortCapture(Key, aborted.Message);
                return false;
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                CudaDecodeDynParams.Deactivate();
                _graphs.AbortCapture(Key, "out of device memory");
                return false;
            }
            catch
            {
                CudaDecodeDynParams.Deactivate();
                _graphs.AbortCapture(Key, "the step threw");
                throw;
            }
            CudaDecodeDynParams.Deactivate();
            if (!ok)
            {
                _graphs.AbortCapture(Key, "the step declined");
                return false;
            }
            try
            {
                Captured = _graphs.EndCaptureAndLaunch(Key, _stable.Values.First(), written);
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                return false;
            }
            return Captured;
        }

        /// <summary>CUDA_ERROR_OUT_OF_MEMORY: a graph the device has no memory left to instantiate or launch.</summary>
        private const int OutOfMemory = 2;

        public bool TryLaunch(ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows)
        {
            if (!Captured) throw new InvalidOperationException("no decode step is recorded");
            _allocator.OnRecordedStepLaunching();
            Upload();
            _dyn.Write(attendLengths, writeRows);
            try
            {
                _graphs.Replay(Key);
            }
            catch (CudaException e) when (e.ErrorCode == OutOfMemory)
            {
                return false;
            }
            _allocator.OnRecordedStepLaunched();
            return true;
        }

        public void MarkWritten(IReadOnlyList<Tensor> tensors)
        {
            foreach (Tensor t in tensors)
                if (t?.Storage is CudaStorage storage) storage.MarkDeviceModified();
        }

        /// <summary>The pinned buffer of the varying inputs, laid out once before the recording (their shapes are fixed from
        /// then on); left unset (each uploaded per launch) where the host cannot pin it.</summary>
        private void PinVaryingInputs()
        {
            if (_pinned != IntPtr.Zero) return;
            long total = 0;
            foreach (int slot in _varying)
                if (_stable.TryGetValue(slot, out Tensor t) && t.Storage is CudaStorage storage)
                {
                    _pinnedAt[slot] = total;
                    total += (storage.ByteLength + 255) / 256 * 256;
                }
            if (total == 0) return;
            _allocator.Context.MakeCurrent();
            if (CudaDriverApi.cuMemHostAlloc(out IntPtr pinned, (UIntPtr)total, 0) != 0)
            {
                _pinnedAt.Clear();
                return;
            }
            _pinned = pinned;
            // An input the host did not write since its last upload: the copy node must deliver what the device holds.
            _allocator.Stream.Synchronize();
            foreach ((int slot, long at) in _pinnedAt)
            {
                var storage = (CudaStorage)_stable[slot].Storage;
                if (!storage.IsHostDirty)
                    CudaDriverApi.cuMemcpyDtoH(_pinned + (nint)at, storage.DeviceBuffer, (UIntPtr)storage.ByteLength).ThrowOnError();
            }
        }

        /// <summary>Inside the recording: a copy node per varying input, from its place in the pinned buffer.</summary>
        private void EnqueueVaryingCopies()
        {
            if (_pinned == IntPtr.Zero) return;
            foreach ((int slot, long at) in _pinnedAt)
            {
                var storage = (CudaStorage)_stable[slot].Storage;
                CudaDriverApi.cuMemcpyHtoDAsync(storage.DeviceBuffer, _pinned + (nint)at, (UIntPtr)storage.ByteLength,
                    _allocator.Stream.Handle).ThrowOnError();
            }
        }

        /// <summary>The stable inputs' host writes go to the device now: the recorded kernels read them there, and an
        /// upload inside the recording would be refused. A varying input's writes go to the pinned buffer instead, which
        /// the recording's copy node reads when the step runs.</summary>
        private void Upload()
        {
            foreach ((int slot, Tensor t) in _stable)
                if (t.Storage is CudaStorage storage)
                {
                    if (_pinned != IntPtr.Zero && _pinnedAt.TryGetValue(slot, out long at))
                    {
                        storage.TakeHostWrites(_pinned + (nint)at);
                        continue;
                    }
                    storage.EnsureDeviceCurrent();
                    // A varying input's host copy is not what later launches read: a host read inside the recording
                    // must synchronize (and so abort it), never bake this launch's values in.
                    if (_varying.Contains(slot)) storage.MarkDeviceModified();
                }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _graphs.Dispose();
            _dyn.Dispose();
            if (_pinned != IntPtr.Zero)
            {
                _allocator.Context.MakeCurrent();
                CudaDriverApi.cuMemFreeHost(_pinned);
                _pinned = IntPtr.Zero;
            }
            foreach (Tensor t in _stable.Values) t.Dispose();
            _stable.Clear();
        }
    }
}
