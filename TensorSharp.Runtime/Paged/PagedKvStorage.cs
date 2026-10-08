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
    /// Physical byte storage for the paged KV pool: <c>numBlocks</c> slabs of
    /// <c>blockByteSize</c> bytes each. Indexed by <see cref="KvBlock.Id"/>.
    ///
    /// The block layout is whatever the model's <c>TryExtractKVBlock</c>
    /// produces. The default uses lazy managed arrays; configured tiered storage
    /// uses budgeted native RAM and SSD with scoped leases. Device KV is still
    /// owned by the model and populated through its synchronous injection contract.
    /// </summary>
    public sealed class PagedKvStorage : IDisposable
    {
        private readonly byte[]?[] _slabs;
        private readonly long _blockByteSize;
        private readonly int _numBlocks;
        private bool _disposed;
        private readonly TieredKvSnapshots? _tiered;

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

            _numBlocks = numBlocks;
            _blockByteSize = blockByteSize;
            _slabs = new byte[numBlocks][];
            if (snapshotOptions != null && blockByteSize > 0)
                _tiered = new TieredKvSnapshots(blockByteSize, snapshotOptions);
            // Lazy slab allocation: many configurations over-provision blocks for
            // worst-case context but only the active prefix actually gets used.
        }

        public int NumBlocks => _numBlocks;
        public long BlockByteSize => _blockByteSize;
        /// <summary>Logical page capacity, not committed RAM. Use MemoryUsage for tiered charges.</summary>
        public long ReservedBytes => (long)_numBlocks * _blockByteSize;
        public bool UsesTieredSnapshots => _tiered != null;
        public IReadOnlyList<MemoryPoolSnapshot>? MemoryUsage => _tiered?.Budget.Snapshot();
        public MemorySchedulerStats? ResidencyStats => _tiered?.Scheduler.GetStats();

        /// <summary>Acquire bytes only for the duration of a synchronous model call.</summary>
        public KvSnapshotLease Acquire(int blockId, ResourceAccess access = ResourceAccess.Read)
        {
            CheckBlock(blockId);
            if (access != ResourceAccess.Read && access != ResourceAccess.Write)
                throw new ArgumentOutOfRangeException(nameof(access));
            if (_tiered != null) return _tiered.Acquire(blockId, access);
            EnsureSlab(blockId);
            return new KvSnapshotLease(_slabs[blockId]!, access);
        }

        internal Span<byte> CaptureScratch(int bytes) => _tiered!.CaptureScratch(bytes);
        internal System.Threading.Tasks.Task<bool>? TryPrefetch(int blockId) => _tiered?.TryPrefetch(blockId);

        /// <summary>Get a writable span for block <paramref name="blockId"/>. Allocates
        /// the slab on first access. The returned span is exactly <see cref="BlockByteSize"/>
        /// long.</summary>
        public Span<byte> GetSpan(int blockId)
        {
            if (_tiered != null) throw new InvalidOperationException("Tiered snapshots require Acquire and a scoped lease.");
            EnsureSlab(blockId);
            return _slabs[blockId].AsSpan();
        }

        /// <summary>Read-only view of block <paramref name="blockId"/>.</summary>
        public ReadOnlySpan<byte> GetReadOnlySpan(int blockId)
        {
            if (_tiered != null) throw new InvalidOperationException("Tiered snapshots require Acquire and a scoped lease.");
            EnsureSlab(blockId);
            return _slabs[blockId].AsSpan();
        }

        /// <summary>Release the managed slab or unregister the tiered resource.
        /// Used when the block is being evicted from the cache and the bytes will
        /// not be reused.</summary>
        public void ReleaseSlab(int blockId)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((uint)blockId >= (uint)_numBlocks) return;
            _tiered?.Release(blockId);
            _slabs[blockId] = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void EnsureSlab(int blockId)
        {
            CheckBlock(blockId);
            if (_slabs[blockId] == null)
                _slabs[blockId] = new byte[_blockByteSize];
        }

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
        }
    }
}
