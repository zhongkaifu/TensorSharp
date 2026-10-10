// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Physical byte storage for the paged KV pool: up to <c>numBlocks</c> slabs of at
    /// most <c>blockByteSize</c> bytes each. Indexed by <see cref="KvBlock.Id"/>.
    ///
    /// The block layout is whatever the model's <c>TryExtractKVBlock</c>
    /// produces - this class just owns the bytes. Storage is in managed memory; for
    /// the GGML/CUDA path the bytes are shuttled into device-resident KV tensors by
    /// the model layer at inject time.
    ///
    /// <para>A slab is normally the full <see cref="BlockByteSize"/>. A per-block-capture
    /// (recurrent) model writes most of its blocks in a smaller K/V-only form
    /// (<c>IModelArchitecture.ComputeKVBlockByteSizeWithoutRecurrentState</c>): such a
    /// slab is allocated at exactly the length written (<see cref="GetSpan(int, int)"/>),
    /// and <see cref="GetReadOnlySpan"/> returns that real length, which is how a reader
    /// tells the two forms apart. <see cref="AllocatedBytes"/> is what the slabs actually
    /// hold; <see cref="ReservedBytes"/> the worst case if every block held a full one.</para>
    /// </summary>
    public sealed class PagedKvStorage : IDisposable
    {
        private readonly byte[]?[] _slabs;
        private readonly long _blockByteSize;
        private readonly int _numBlocks;
        private long _allocatedBytes;
        private bool _disposed;

        public PagedKvStorage(int numBlocks, long blockByteSize)
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
            // Lazy slab allocation: many configurations over-provision blocks for
            // worst-case context but only the active prefix actually gets used.
        }

        public int NumBlocks => _numBlocks;

        /// <summary>The full (largest) slab size: one block of every layer's K/V and,
        /// for a recurrent model, its running state.</summary>
        public long BlockByteSize => _blockByteSize;

        /// <summary>Upper bound: every block holding a full slab.</summary>
        public long ReservedBytes => (long)_numBlocks * _blockByteSize;

        /// <summary>Bytes the allocated slabs actually hold right now.</summary>
        public long AllocatedBytes => _allocatedBytes;

        /// <summary>Get a writable full-size span for block <paramref name="blockId"/>.
        /// Allocates the slab on first access, and replaces a shorter slab written in the
        /// K/V-only form. The returned span is exactly <see cref="BlockByteSize"/> long.</summary>
        public Span<byte> GetSpan(int blockId)
        {
            CheckId(blockId);
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
            CheckId(blockId);
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
            CheckId(blockId);
            byte[]? slab = _slabs[blockId] ?? Replace(blockId, _blockByteSize);
            return slab.AsSpan();
        }

        /// <summary>Bytes block <paramref name="blockId"/>'s slab holds; 0 when none is
        /// allocated.</summary>
        public long SlabLength(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks) return 0;
            return _slabs[blockId]?.LongLength ?? 0;
        }

        /// <summary>Drop the slab for block <paramref name="blockId"/> back to the GC.
        /// Used when the block is being evicted from the cache and the bytes will
        /// not be reused.</summary>
        public void ReleaseSlab(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks) return;
            byte[]? slab = _slabs[blockId];
            if (slab == null) return;
            _allocatedBytes -= slab.LongLength;
            _slabs[blockId] = null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void CheckId(int blockId)
        {
            if ((uint)blockId >= (uint)_numBlocks)
                throw new ArgumentOutOfRangeException(nameof(blockId), $"Block id {blockId} out of range [0,{_numBlocks}).");
        }

        private byte[] Replace(int blockId, long length)
        {
            byte[]? old = _slabs[blockId];
            if (old != null) _allocatedBytes -= old.LongLength;
            var slab = new byte[length];
            _slabs[blockId] = slab;
            _allocatedBytes += length;
            return slab;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            for (int i = 0; i < _slabs.Length; i++)
                _slabs[i] = null;
            _allocatedBytes = 0;
        }
    }
}
