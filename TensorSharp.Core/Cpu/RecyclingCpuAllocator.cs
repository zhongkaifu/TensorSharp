// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// A CPU allocator that keeps the large buffers its storages free and hands them to the next storage of the same size.
    /// A graph program that runs again and again (a diffusion denoiser's steps over one image) frees and allocates the same
    /// intermediates every run: with plain allocations each large one is a fresh mapping whose pages fault in again on every
    /// step (about 1.5 GB a step of Qwen-Image-2.1 at 256 rows), where recycled ones are written in place. Buffers below
    /// <see cref="MinRecycledBytes"/> go to the C allocator as before; at most <see cref="MaxRetainedBytes"/> of free buffers
    /// are kept (the least recently freed go first), and <see cref="Trim"/> returns them all.
    /// </summary>
    public sealed class RecyclingCpuAllocator : CpuAllocator, IDisposable
    {
        /// <summary>Buffers smaller than this are not kept (the C allocator serves them from its own free lists).</summary>
        public const long MinRecycledBytes = 256 * 1024;

        private readonly object _gate = new();
        private readonly Dictionary<long, Stack<IntPtr>> _free = new();
        private readonly LinkedList<(IntPtr Buffer, long Bytes)> _order = new();
        private long _retained;

        public RecyclingCpuAllocator(BlasEnum blasEnum, long maxRetainedBytes) : base(blasEnum)
        {
            if (maxRetainedBytes < 0) throw new ArgumentOutOfRangeException(nameof(maxRetainedBytes));
            MaxRetainedBytes = maxRetainedBytes;
        }

        /// <summary>The most free bytes kept for reuse.</summary>
        public long MaxRetainedBytes { get; }

        /// <summary>Free bytes kept now.</summary>
        public long RetainedBytes
        {
            get { lock (_gate) return _retained; }
        }

        internal IntPtr Rent(long bytes)
        {
            if (bytes >= MinRecycledBytes)
                lock (_gate)
                    if (_free.TryGetValue(bytes, out Stack<IntPtr> stack) && stack.Count > 0)
                    {
                        IntPtr buffer = stack.Pop();
                        _retained -= bytes;
                        for (LinkedListNode<(IntPtr Buffer, long Bytes)> n = _order.Last; n != null; n = n.Previous)
                            if (n.Value.Buffer == buffer)
                            {
                                _order.Remove(n);
                                break;
                            }
                        return buffer;
                    }
            return Marshal.AllocHGlobal(new IntPtr(bytes));
        }

        internal void Return(IntPtr buffer, long bytes)
        {
            if (buffer == IntPtr.Zero) return;
            if (bytes < MinRecycledBytes || bytes > MaxRetainedBytes)
            {
                Marshal.FreeHGlobal(buffer);
                return;
            }
            lock (_gate)
            {
                // Make room: the buffers freed longest ago go back to the C allocator.
                while (_retained + bytes > MaxRetainedBytes && _order.First != null)
                {
                    var (old, oldBytes) = _order.First.Value;
                    _order.RemoveFirst();
                    Stack<IntPtr> stack = _free[oldBytes];
                    var rest = new Stack<IntPtr>();
                    while (stack.Count > 0)
                    {
                        IntPtr b = stack.Pop();
                        if (b != old) rest.Push(b);
                    }
                    while (rest.Count > 0) stack.Push(rest.Pop());
                    _retained -= oldBytes;
                    Marshal.FreeHGlobal(old);
                }
                if (!_free.TryGetValue(bytes, out Stack<IntPtr> bucket)) _free[bytes] = bucket = new Stack<IntPtr>();
                bucket.Push(buffer);
                _order.AddLast((buffer, bytes));
                _retained += bytes;
            }
        }

        /// <summary>Return every kept buffer to the C allocator (a phase that ends, a released workspace).</summary>
        public void Trim()
        {
            lock (_gate)
            {
                foreach (Stack<IntPtr> stack in _free.Values)
                    while (stack.Count > 0) Marshal.FreeHGlobal(stack.Pop());
                _free.Clear();
                _order.Clear();
                _retained = 0;
            }
        }

        public void Dispose() => Trim();
    }
}
