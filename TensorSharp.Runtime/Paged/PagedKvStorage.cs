// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;
using System.Collections.Generic;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Physical byte storage for the paged KV pool: up to <c>numBlocks</c> slabs of at
    /// most <c>blockByteSize</c> bytes each. Indexed by <see cref="KvBlock.Id"/>.
    ///
    /// The block layout is whatever the model's <c>TryExtractKVBlock</c>
    /// produces. Storage uses lazy managed arrays or budgeted native RAM and SSD
    /// with scoped leases. Device KV remains owned by the model.
    ///
    /// <para>A slab is normally the full <see cref="BlockByteSize"/>. A per-block-capture
    /// (recurrent) model writes most of its blocks in a smaller K/V-only form
    /// (<c>IModelArchitecture.ComputeKVBlockByteSizeWithoutRecurrentState</c>): such a
    /// slab is allocated at exactly the length written (<see cref="GetSpan(int, int)"/>),
    /// and <see cref="GetReadOnlySpan"/> returns that real length, which is how a reader
    /// tells the two forms apart. <see cref="AllocatedBytes"/> is what the slabs actually
    /// hold; <see cref="ReservedBytes"/> the worst case if every block held a full one.
    /// Tiered pages retain full physical capacity, reported by <see cref="SlabLength"/>,
    /// while read leases expose only the published payload length.</para>
    /// </summary>
    public sealed class PagedKvStorage : IDisposable
    {
        private readonly byte[]?[] _slabs;
        private readonly long _blockByteSize;
        private readonly int _numBlocks;
        private long _allocatedBytes;
        private bool _disposed;
        private readonly TieredKvSnapshots? _tiered;
        private readonly int[]? _payloadLengths;

        public PagedKvStorage(int numBlocks, long blockByteSize) : this(numBlocks, blockByteSize, null) { }

        public PagedKvStorage(int numBlocks, long blockByteSize, KvSnapshotOptions? snapshotOptions)
        {
            if (numBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBlocks));
            // blockByteSize == 0 is a metadata-only pool: the model cannot
            // snapshot its KV state into host bytes (e.g. a tensor-parallel
            // model whose KV cache lives sharded across per-rank device
            // backends), but the scheduler still needs the block ids for its
            // block tables. Every byte-level caller (capture/inject in
            // BatchExecutor) is gated on SupportsKVStateSnapshot, so the
            // zero-length slabs are never read or written.
            if (blockByteSize < 0) throw new ArgumentOutOfRangeException(nameof(blockByteSize));
            if (snapshotOptions != null && blockByteSize == 0)
                throw new NotSupportedException("Bounded KV snapshots require a model with a nonempty, restorable block snapshot.");

            _numBlocks = numBlocks;
            _blockByteSize = blockByteSize;
            _slabs = new byte[numBlocks][];
            if (snapshotOptions != null && blockByteSize > 0)
            {
                _payloadLengths = new int[numBlocks];
                Array.Fill(_payloadLengths, -1);
                _tiered = new TieredKvSnapshots(blockByteSize, snapshotOptions);
            }
            // Lazy slab allocation: many configurations over-provision blocks for
            // worst-case context but only the active prefix actually gets used.
        }

        public int NumBlocks => _numBlocks;

        /// <summary>The full (largest) slab size: one block of every layer's K/V and,
        /// for a recurrent model, its running state.</summary>
        public long BlockByteSize => _blockByteSize;

        /// <summary>Upper bound: every block holding a full slab.</summary>
        public long ReservedBytes => (long)_numBlocks * _blockByteSize;
        public bool UsesTieredSnapshots => _tiered != null;
        internal MemoryBudget? Budget => _tiered?.Budget;
        /// <summary>Budget counters. With a shared budget, includes every owner
        /// of its pools, not only this storage instance.</summary>
        public IReadOnlyList<MemoryPoolSnapshot>? MemoryUsage => _tiered?.Budget.Snapshot();
        public MemorySchedulerStats? ResidencyStats => _tiered?.Scheduler.GetStats();

        /// <summary>Acquire bytes only for the duration of a synchronous model call.</summary>
        public KvSnapshotLease Acquire(int blockId, ResourceAccess access = ResourceAccess.Read)
            => Acquire(blockId, access, null);

        public KvSnapshotLease Acquire(int blockId, ResourceAccess access, BudgetReservation? allocationEnvelope)
        {
            CheckBlock(blockId);
            if (access != ResourceAccess.Read && access != ResourceAccess.Write)
                throw new ArgumentOutOfRangeException(nameof(access));
            if (_tiered != null)
            {
                var lease = _tiered.Acquire(blockId, access, allocationEnvelope);
                if (access == ResourceAccess.Write || _payloadLengths![blockId] < 0)
                    _payloadLengths![blockId] = checked((int)_blockByteSize);
                else
                    lease.LimitReadLength(_payloadLengths[blockId]);
                return lease;
            }
            if (allocationEnvelope != null) throw new InvalidOperationException("Request envelopes require budgeted KV snapshots.");
            if (access == ResourceAccess.Write) GetSpan(blockId);
            else GetReadOnlySpan(blockId);
            return new KvSnapshotLease(_slabs[blockId]!, access);
        }

        internal Span<byte> CaptureScratch(int bytes) => _tiered!.CaptureScratch(bytes);

        /// <summary>Publish a freshly captured page. Old spilled bytes are not
        /// needed for replacement; padding is cleared so the entire new page is
        /// initialized, including when the model captured only a partial tail.</summary>
        internal void Store(int blockId, ReadOnlySpan<byte> bytes, BudgetReservation? allocationEnvelope,
            bool exactLength = false)
        {
            CheckBlock(blockId);
            if (bytes.Length > _blockByteSize) throw new ArgumentOutOfRangeException(nameof(bytes));
            if (_tiered != null)
            {
                using var lease = _tiered.Acquire(blockId, ResourceAccess.Write, allocationEnvelope, overwrite: true);
                bytes.CopyTo(lease.Span);
                lease.Span[bytes.Length..].Clear();
                _payloadLengths![blockId] = exactLength ? bytes.Length : checked((int)_blockByteSize);
            }
            else
            {
                if (allocationEnvelope != null) throw new InvalidOperationException("Request envelopes require budgeted KV snapshots.");
                var destination = exactLength ? GetSpan(blockId, bytes.Length) : GetSpan(blockId);
                bytes.CopyTo(destination);
                destination[bytes.Length..].Clear();
            }
        }
        internal System.Threading.Tasks.Task<bool>? TryPrefetch(int blockId, BudgetReservation? allocationEnvelope = null)
            => _tiered?.TryPrefetch(blockId, allocationEnvelope);

        /// <summary>Managed slab bytes. Tiered physical charges are in MemoryUsage.</summary>
        public long AllocatedBytes => _allocatedBytes;

        /// <summary>Get a writable full-size span for block <paramref name="blockId"/>.
        /// Allocates the slab on first access, and replaces a shorter slab written in the
        /// K/V-only form. The returned span is exactly <see cref="BlockByteSize"/> long.</summary>
        public Span<byte> GetSpan(int blockId)
        {
            CheckBlock(blockId);
            if (_tiered != null) throw new InvalidOperationException("Tiered snapshots require Acquire and a scoped lease.");
            byte[]? slab = _slabs[blockId];
            if (slab == null || slab.LongLength != _blockByteSize)
                slab = Replace(blockId, _blockByteSize);
            return slab.AsSpan();
        }

        /// <summary>Get a writable span of exactly <paramref name="length"/> bytes for block
        /// <paramref name="blockId"/>, (re)allocating the slab at that length when it is
        /// missing or of another length. <paramref name="length"/> is at most
        /// <see cref="BlockByteSize"/>.</summary>
        public Span<byte> GetSpan(int blockId, int length)
        {
            CheckBlock(blockId);
            if (_tiered != null) throw new InvalidOperationException("Tiered snapshots require Acquire and a scoped lease.");
            if (length < 0 || length > _blockByteSize)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Slab length {length} is outside [0,{_blockByteSize}].");
            byte[]? slab = _slabs[blockId];
            if (slab == null || slab.Length != length)
                slab = Replace(blockId, length);
            return slab.AsSpan();
        }

        /// <summary>Read-only view of block <paramref name="blockId"/>, at the length it was
        /// written with (a never-written block reads as a zeroed full-size slab).</summary>
        public ReadOnlySpan<byte> GetReadOnlySpan(int blockId)
        {
            CheckBlock(blockId);
            if (_tiered != null) throw new InvalidOperationException("Tiered snapshots require Acquire and a scoped lease.");
            byte[]? slab = _slabs[blockId] ?? Replace(blockId, _blockByteSize);
            return slab.AsSpan();
        }

        /// <summary>Physical page capacity, independent of its logical payload length;
        /// 0 when no page is allocated. Tiered residency charges are in MemoryUsage.</summary>
        public long SlabLength(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks) return 0;
            return _tiered != null ? (_payloadLengths![blockId] < 0 ? 0 : _blockByteSize)
                : _slabs[blockId]?.LongLength ?? 0;
        }

        /// <summary>Release the managed slab or unregister the tiered resource.
        /// Used when the block is being evicted from the cache and the bytes will
        /// not be reused.</summary>
        public void ReleaseSlab(int blockId)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((uint)blockId >= (uint)_numBlocks) return;
            _tiered?.Release(blockId); // Retain metadata when a live lease prevents release.
            if (_payloadLengths != null) _payloadLengths[blockId] = -1;
            byte[]? slab = _slabs[blockId];
            if (slab == null) return;
            _allocatedBytes -= slab.LongLength;
            _slabs[blockId] = null;
        }

        private byte[] Replace(int blockId, long length)
        {
            byte[]? old = _slabs[blockId];
            var slab = new byte[length]; // Allocation failure leaves the old slab and ledger intact.
            _slabs[blockId] = slab;
            _allocatedBytes += length - (old?.LongLength ?? 0);
            return slab;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckBlock(int blockId)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((uint)blockId >= (uint)_numBlocks)
                throw new ArgumentOutOfRangeException(nameof(blockId), $"Block id {blockId} out of range [0,{_numBlocks}).");
        }

        public void Dispose()
        {
            if (_disposed) return;
            _tiered?.Dispose();
            _disposed = true;
            for (int i = 0; i < _slabs.Length; i++)
                _slabs[i] = null;
            _allocatedBytes = 0;
            if (_payloadLengths != null) Array.Fill(_payloadLengths, -1);
        }
    }
}
