// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public sealed partial class NemotronRuntimeModel
    {
        /// <summary>
        /// One rank's Mamba-2 state of one layer for one sequence, as tensors the graphs bind as state leaves: the conv window
        /// [channels, kernel - 1] (oldest first per channel) and the scan state [heads, head_dim, d_state]. A verify window keeps
        /// its rows' states ([K slots of the scan state], newest first) and its conv inputs ([channels, kernel - 1 + rows]) apart,
        /// and the live state stays the one before it until <see cref="CommitRecurrentPrefix"/> takes the accepted prefix's.
        /// Where the executor binds host bytes (GGML) the tensors are host-backed and the device copy is authoritative: a reset
        /// zeroes the host bytes and drops (or rewrites) the device copy; elsewhere they are the backend's own tensors.
        /// </summary>
        internal sealed class Mamba2State : RecurrentState
        {
            private readonly Func<Tensor, IntPtr> _hostPointer;
            private readonly IGraphExecutor _executor;
            private readonly IAllocator _allocator;
            private readonly bool _host;

            public Mamba2State(IAllocator allocator, IGraphExecutor executor, Func<Tensor, IntPtr> hostPointer, int convRows, int channels,
                int heads, int headDim, int stateSize)
            {
                _allocator = allocator;
                _executor = executor;
                _hostPointer = hostPointer;
                _host = executor?.Capabilities.BindsHostPointers ?? true;
                ConvRows = Math.Max(1, convRows);
                Channels = channels;
                StateFloats = (long)heads * headDim * stateSize;
                Conv = new Tensor(allocator, DType.Float32, channels, ConvRows);
                Ssm = new Tensor(allocator, DType.Float32, heads, headDim, stateSize);
                // A fresh mapping is zero already: clearing it would only fault every page in.
                foreach (Tensor t in new[] { Conv, Ssm })
                {
                    if (_host && t.Storage.TakeAllocationZeroGuarantee()) Invalidate(t);
                    else Clear(t);
                }
            }

            public Tensor Conv { get; }
            public Tensor Ssm { get; }
            public int ConvRows { get; }
            public int Channels { get; }
            public long StateFloats { get; }

            /// <summary>A verify window's states, newest first: room for <see cref="SnapshotCapacity"/> rows.</summary>
            public Tensor Verify { get; private set; }

            /// <summary>A verify window's conv inputs [channels, ConvRows + rows] at the window's own pitch (room for ConvRows +
            /// capacity a channel): columns [k, k + ConvRows) are the window after k of its rows.</summary>
            public Tensor ConvHistory { get; private set; }

            public int SnapshotCapacity { get; private set; }

            /// <summary>Room for a <paramref name="rows"/>-row verify; true when the buffers moved.</summary>
            public bool EnsureSpecBuffers(int rows)
            {
                if (rows <= SnapshotCapacity) return false;
                int capacity = 4;
                while (capacity < rows) capacity *= 2;
                Free(Verify);
                Free(ConvHistory);
                Verify = new Tensor(_allocator, DType.Float32, capacity * StateFloats);
                ConvHistory = new Tensor(_allocator, DType.Float32, Channels, ConvRows + capacity);
                SnapshotCapacity = capacity;
                NoteBindingsChanged();
                return true;
            }

            protected internal override void ReleaseSpecBuffers(bool keepInPlaceBuffers)
            {
                if (Verify == null && ConvHistory == null) return;
                NoteBindingsChanged();
                Free(Verify);
                Free(ConvHistory);
                Verify = null;
                ConvHistory = null;
                SnapshotCapacity = 0;
            }

            protected internal override long StateBytes
                => Conv.Storage.ByteLength + Ssm.Storage.ByteLength + SpecBufferBytes;

            protected internal override long SpecBufferBytes
                => (Verify?.Storage.ByteLength ?? 0) + (ConvHistory?.Storage.ByteLength ?? 0);

            public override void Reset()
            {
                Clear(Conv);
                Clear(Ssm);
            }

            /// <summary>Zero the host bytes and write them into each existing device copy: nothing a cached graph binds is freed.</summary>
            public override bool ResetInPlace()
            {
                if (!_host)
                {
                    Reset();
                    return true;
                }
                ClearInPlace(Conv);
                ClearInPlace(Ssm);
                return true;
            }

            protected internal override bool CopyFromInPlace(RecurrentState source)
            {
                if (!_host)
                {
                    CopyFrom(source);
                    return true;
                }
                var other = (Mamba2State)source;
                CopyInPlace(other.Conv, Conv);
                CopyInPlace(other.Ssm, Ssm);
                return true;
            }

            protected internal override IEnumerable<Tensor> HostTensors => new[] { Conv, Ssm };

            protected internal override bool CanCopy => true;

            protected internal override bool GraphTensorsResetToZero => true;

            protected internal override IEnumerable<Tensor> CheckpointTensors => new[] { Conv, Ssm };

            protected internal override Tensor[] GraphTensors => _host ? new[] { Conv, Ssm } : null;

            protected internal override unsafe void CopyFrom(RecurrentState source)
            {
                var other = (Mamba2State)source;
                foreach (var (from, to) in new[] { (other.Conv, Conv), (other.Ssm, Ssm) })
                {
                    if (!_host)
                    {
                        Ops.Copy(to, from);
                        continue;
                    }
                    long bytes = to.ElementCount() * sizeof(float);
                    Buffer.MemoryCopy((void*)_hostPointer(from), (void*)_hostPointer(to), bytes, bytes);
                }
                OnHostWritten();
            }

            protected internal override void OnHostWritten()
            {
                if (!_host) return;
                Invalidate(Conv);
                Invalidate(Ssm);
            }

            private unsafe void ClearInPlace(Tensor t)
            {
                long bytes = t.ElementCount() * sizeof(float);
                IntPtr p = _hostPointer(t);
                new Span<float>((float*)p, checked((int)t.ElementCount())).Clear();
                GgmlBasicOps.UploadHostBufferRanges(p, stackalloc long[] { 0 }, stackalloc long[] { bytes });
            }

            private unsafe void CopyInPlace(Tensor from, Tensor to)
            {
                long bytes = to.ElementCount() * sizeof(float);
                Buffer.MemoryCopy((void*)_hostPointer(from), (void*)_hostPointer(to), bytes, bytes);
                GgmlBasicOps.UploadHostBufferRanges(_hostPointer(to), stackalloc long[] { 0 }, stackalloc long[] { bytes });
            }

            private void Invalidate(Tensor t)
            {
                _executor?.Invalidate(_hostPointer(t));
                GgmlBasicOps.InvalidateTensorHostBuffer(t);
            }

            /// <summary>Dispose <paramref name="t"/> (null: nothing) with the device copies graphs made of its host bytes.</summary>
            private void Free(Tensor t)
            {
                if (t == null) return;
                if (_host) Invalidate(t);
                t.Dispose();
            }

            private unsafe void Clear(Tensor t)
            {
                if (!_host)
                {
                    Ops.Fill(t, 0);
                    return;
                }
                IntPtr p = _hostPointer(t);
                new Span<float>((float*)p, checked((int)t.ElementCount())).Clear();
                Invalidate(t);
            }

            public override void Dispose()
            {
                Free(Conv);
                Free(Ssm);
                Free(Verify);
                Free(ConvHistory);
            }
        }
    }
}
