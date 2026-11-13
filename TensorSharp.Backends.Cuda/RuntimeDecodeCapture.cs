// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
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
            Upload();
            _dyn.Write(attendLength, writeRow, 0, 0);
            if (!_graphs.BeginCapture(Key)) return false;
            bool ok;
            try
            {
                _dyn.EnqueueUpload();
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

        /// <summary>The stable inputs' host writes go to the device now: the recorded kernels read them there, and an
        /// upload inside the recording would be refused.</summary>
        private void Upload()
        {
            foreach ((int slot, Tensor t) in _stable)
                if (t.Storage is CudaStorage storage)
                {
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
            foreach (Tensor t in _stable.Values) t.Dispose();
            _stable.Clear();
        }
    }
}
