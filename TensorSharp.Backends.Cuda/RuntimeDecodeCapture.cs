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
    /// host. The recording owns the pool blocks its kernels use (<see cref="CudaPrefillGraphCache"/>); the attention length
    /// and the cache write row come from a dynamic-parameter block the graph uploads first (<see cref="CudaDecodeDynParams"/>),
    /// which the decode attention and the one-row cache append read on the device.
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
            _graphs = new CudaPrefillGraphCache(allocator);
            _dyn = new CudaDecodeDynParams(allocator);
        }

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

        public bool CaptureAndLaunch(Func<bool> run, int attendLength, int writeRow, IReadOnlyList<Tensor> written)
        {
            if (Captured || _disposed || !_graphs.IsUsable || !_dyn.IsValid || _stable.Count == 0) return false;
            _allocator.OnRecordedStepLaunching();
            PinVaryingInputs();
            Upload();
            _dyn.Write(attendLength, writeRow, 0, 0);
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
            Captured = _graphs.EndCaptureAndLaunch(Key, _stable.Values.First(), written);
            return Captured;
        }

        public void Launch(int attendLength, int writeRow)
        {
            if (!Captured) throw new InvalidOperationException("no decode step is recorded");
            _allocator.OnRecordedStepLaunching();
            Upload();
            _dyn.Write(attendLength, writeRow, 0, 0);
            _graphs.Replay(Key);
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
