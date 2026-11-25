// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Recycling of the managed (pure-C#) VAE's feature maps within one encode or decode.
//
// Why: a decoder feature map is hundreds of MB to several GB (288 channels at 2048x2048 is
// 4.8 GB), and each one lives for a single layer. Allocated fresh, the dead maps stayed
// committed until the GC got round to a gen-2 collection: a 1024x1024 decode peaked at 13.5 GB
// of working set where at most 2.3 GB of maps are ever live, and a 2048x2048 decode would not
// have fit in 32 GB. The drivers in QwenImage21Vae / VaeReferenceMath release every map the
// moment its last reader is done (Release), and Feature.Uninitialized rents from here, so a map
// is reused by the next layer that needs one of exactly that size (the decoder's map sizes
// repeat within and across stages). What cannot be reused is dropped and collected before the
// heap grows.
//
// Only the managed fast path rents (VaeReferenceMath.FastCpu); the device and scalar paths
// allocate as before, and Release is a no-op without an active pool. Every renter overwrites
// all of its buffer (Feature.Uninitialized contract), so recycling never changes a result:
// pooled and unpooled decodes are bit-identical (tested with released buffers poisoned).
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class VaeFeaturePool : IDisposable
    {
        // Settable by tests and the stages bench (--no-pool): off, every map is a fresh
        // allocation that the GC reclaims on its own schedule.
        internal static bool Enabled { get; set; } = true;

        /// <summary>Test hook: fill every returned buffer with NaN, so a map read through a stale
        /// array reference after its release poisons the output instead of reading recycled data.</summary>
        internal static bool PoisonReturned { get; set; }

        // An aggressive collection also decommits what it frees. A plain forced one leaves the
        // freed regions committed for reuse, but the next map is usually a different size and
        // gets fresh memory next to them. Measured at 2048x2048: peak commit 12.8 instead of
        // 14.0 GB at the same decode time; after a 1024x1024 decode 1.2 GB instead of 4.0 GB
        // stayed committed.

        // Dropped (unreusable) buffers larger than this are collected before the heap grows.
        private const long CollectThresholdBytes = 64L << 20;

        [ThreadStatic] private static VaeFeaturePool t_current;

        /// <summary>The last pool disposed on this thread (its counters), for tests and benches.</summary>
        [ThreadStatic] internal static VaeFeaturePool LastCompleted;

        private readonly VaeFeaturePool _outer;
        private readonly Dictionary<int, Stack<float[]>> _free = new();
        private readonly HashSet<float[]> _freeSet = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<float[]> _rented = new(ReferenceEqualityComparer.Instance);
        private long _freeBytes, _liveBytes, _droppedBytes;
        private bool _disposed;

        /// <summary>Largest total of rented, not yet returned maps (the drivers' live set).</summary>
        internal long PeakLiveBytes { get; private set; }
        /// <summary>Largest total of rented plus pooled (free) maps: what the pool kept reachable.</summary>
        internal long PeakHeldBytes { get; private set; }
        internal long Rents { get; private set; }
        internal long Hits { get; private set; }
        internal long Collections { get; private set; }
        /// <summary>Rented maps never released when the pool closed (the drivers release every
        /// map, the output included, so this is 0; tests assert it).</summary>
        internal long LiveBytesAtDispose { get; private set; }

        private VaeFeaturePool(VaeFeaturePool outer) => _outer = outer;

        /// <summary>The pool of the encode/decode running on this thread, or null.</summary>
        internal static VaeFeaturePool Current => t_current;

        /// <summary>Starts recycling on this thread until the returned pool is disposed, or returns
        /// null (nothing to dispose) when <paramref name="active"/> is false or the pool is off.</summary>
        internal static VaeFeaturePool Enter(bool active)
        {
            if (!active || !Enabled) return null;
            var pool = new VaeFeaturePool(t_current);
            t_current = pool;
            return pool;
        }

        /// <summary>An uninitialized buffer of exactly <paramref name="length"/> floats: a pooled one
        /// of that length when there is one, else a new allocation.</summary>
        internal float[] Rent(int length)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Rents++;
            float[] buffer;
            bool hit = _free.TryGetValue(length, out var stack) && stack.Count > 0;
            if (hit)
            {
                buffer = stack.Pop();
                _freeSet.Remove(buffer);
                _freeBytes -= Bytes(buffer);
                Hits++;
            }
            else
            {
                // A miss grows the heap. The pooled buffers of other lengths belong to layers that
                // have passed (the decoder moves on to larger maps), so drop them first and, when
                // that is a lot of memory, collect it now: the GC would otherwise leave it
                // committed next to the new map until its own next gen-2 collection.
                DropFree();
                CollectDropped(force: false);
                try { buffer = GC.AllocateUninitializedArray<float>(length); }
                catch (OutOfMemoryException)
                {
                    CollectDropped(force: true);
                    buffer = GC.AllocateUninitializedArray<float>(length);
                }
            }
            _rented.Add(buffer);
            _liveBytes += Bytes(buffer);
            PeakLiveBytes = Math.Max(PeakLiveBytes, _liveBytes);
            PeakHeldBytes = Math.Max(PeakHeldBytes, _liveBytes + _freeBytes);
            return buffer;
        }

        /// <summary>Takes back a buffer nobody reads any more. Buffers the pool did not rent (the
        /// decoder's input, zero-initialized maps) are adopted for reuse too.</summary>
        internal void Return(float[] buffer)
        {
            if (buffer == null || _disposed) return;
            if (_freeSet.Contains(buffer))
                throw new InvalidOperationException("A VAE feature map was released twice.");
            if (_rented.Remove(buffer)) _liveBytes -= Bytes(buffer);
            if (PoisonReturned) buffer.AsSpan().Fill(float.NaN);
            if (!_free.TryGetValue(buffer.Length, out var stack)) _free[buffer.Length] = stack = new Stack<float[]>();
            stack.Push(buffer);
            _freeSet.Add(buffer);
            _freeBytes += Bytes(buffer);
            PeakHeldBytes = Math.Max(PeakHeldBytes, _liveBytes + _freeBytes);
        }

        /// <summary>Returns <paramref name="feature"/>'s storage to this thread's pool (no-op
        /// without one) and detaches it from the feature (D becomes null). The caller must not
        /// read the feature afterwards. Detaching matters for memory, not only for safety: the
        /// drivers' locals still reference released features (a stage input across its residual
        /// blocks, the argument of x = Residual21(x)), and a buffer the pool drops must be
        /// unreachable for the collection before the next allocation to free it. Measured at
        /// 2048x2048: without it the 4.6 GB stage-4 input outlived its last read.</summary>
        internal static void Release(Feature feature)
        {
            var pool = t_current;
            if (feature == null || pool == null) return;
            float[] data = feature.D ?? throw new InvalidOperationException("A VAE feature map was released twice.");
            feature.D = null;
            pool.Return(data);
        }

        /// <summary>One-line summary of the counters, for tests and benches.</summary>
        internal string Describe() =>
            $"feature pool: {Rents} maps, {Hits} reused, peak live {PeakLiveBytes / 1048576.0:F0} MiB, " +
            $"peak held {PeakHeldBytes / 1048576.0:F0} MiB, {Collections} collection(s)";

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (t_current == this) t_current = _outer;
            LastCompleted = this;
            LiveBytesAtDispose = _liveBytes;
            // Everything still pooled is garbage now. A multi-GB decode leaves GBs of it that
            // nothing else in the process would collect soon (the transformer and the encoders
            // allocate natively), so collect (and decommit) it here.
            _droppedBytes += _liveBytes;
            DropFree();
            _rented.Clear();
            _liveBytes = 0;
            if (_droppedBytes >= CollectThresholdBytes || Collections > 0)
            {
                _droppedBytes = 0;
                Collections++;
                Collect();
            }
        }

        private void DropFree()
        {
            _droppedBytes += _freeBytes;
            _free.Clear();
            _freeSet.Clear();
            _freeBytes = 0;
        }

        private void CollectDropped(bool force)
        {
            if (!force && _droppedBytes < CollectThresholdBytes) return;
            _droppedBytes = 0;
            Collections++;
            Collect();
        }

        private static void Collect()
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
        }

        private static long Bytes(float[] buffer) => (long)buffer.Length * sizeof(float);
    }
}
