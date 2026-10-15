// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Owner of all physical KV blocks. Mirrors vLLM's <c>BlockPool</c>: a fixed-
    /// size array of <see cref="KvBlock"/> metadata and a free queue. Prefix reuse
    /// holds blocks through references of its own (the radix prefix cache).
    ///
    /// The pool is not thread-safe; the scheduler is the single owner.
    /// </summary>
    public sealed class BlockPool
    {
        private readonly KvBlock[] _blocks;
        private readonly FreeBlockQueue _freeQueue;
        private readonly PagedKvStorage _storage;
        private readonly int _blockSize;

        public BlockPool(int numBlocks, int blockSize, long blockByteSize) : this(numBlocks, blockSize, blockByteSize, null) { }

        public BlockPool(int numBlocks, int blockSize, long blockByteSize, KvSnapshotOptions? snapshotOptions)
        {
            if (numBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBlocks));
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));

            _blockSize = blockSize;
            _blocks = new KvBlock[numBlocks];
            _freeQueue = new FreeBlockQueue();
            _storage = new PagedKvStorage(numBlocks, blockByteSize, snapshotOptions);

            for (int i = 0; i < numBlocks; i++)
            {
                _blocks[i] = new KvBlock(i);
                _freeQueue.Enqueue(_blocks[i]);
            }
        }

        public int BlockSize => _blockSize;
        public int NumBlocks => _blocks.Length;
        public int NumFreeBlocks => _freeQueue.Count;
        public PagedKvStorage Storage => _storage;

        /// <summary>Find a block by physical id. Used by the executor when it needs
        /// to look up storage bytes during inject/extract.</summary>
        public KvBlock GetBlock(int id) => _blocks[id];

        /// <summary>Bump the ref count of a block. Used when the prefix cache holds a
        /// block or a sequence adopts one it holds.</summary>
        public void Touch(KvBlock block)
        {
            if (block.RefCount == 0)
                _freeQueue.Remove(block);
            block.RefCount++;
        }

        /// <summary>Allocate <paramref name="count"/> empty blocks from the free
        /// queue. Returns null when the pool is exhausted (the scheduler will
        /// then preempt). Each returned block has RefCount=1, Used=0.</summary>
        public KvBlock[]? AllocateNew(int count)
        {
            if (count <= 0) return Array.Empty<KvBlock>();
            if (_freeQueue.Count < count) return null;

            var result = new KvBlock[count];
            for (int i = 0; i < count; i++)
            {
                KvBlock block = _freeQueue.Dequeue()
                    ?? throw new InvalidOperationException("The free queue became empty during allocation.");
                block.RefCount = 1;
                block.Used = 0;
                block.IsRestorablePrefixEnd = true;
                // A new owner rewrites the block from position 0 on whichever path it
                // takes; neither the previous paged K/V nor its snapshot describes it.
                block.HoldsModelPagedKv = false;
                block.HoldsSnapshotBytes = false;
                result[i] = block;
            }
            return result;
        }

        /// <summary>Decrement the ref count of each block. Blocks whose ref count
        /// hits zero return to the back of the free queue and drop their bytes.</summary>
        public void Free(IReadOnlyList<KvBlock?>? blocks)
        {
            if (blocks == null) return;
            for (int i = 0; i < blocks.Count; i++)
            {
                KvBlock? b = blocks[i];
                if (b == null) continue;
                if (b.RefCount <= 0)
                    throw new InvalidOperationException($"Double-free of block {b.Id}");
                if (b.RefCount == 1) ReleaseStorage(b);
                b.RefCount--;
                if (b.RefCount == 0)
                {
                    _freeQueue.Enqueue(b);
                }
            }
        }

        /// <summary>Free a single block.</summary>
        public void Free(KvBlock? block)
        {
            if (block == null) return;
            if (block.RefCount <= 0)
                throw new InvalidOperationException($"Double-free of block {block.Id}");
            if (block.RefCount == 1) ReleaseStorage(block);
            block.RefCount--;
            if (block.RefCount == 0)
            {
                _freeQueue.Enqueue(block);
            }
        }

        private void ReleaseStorage(KvBlock block)
        {
            // The prefix cache owns a reference while it caches a page, so a block
            // whose last reference is gone holds nothing anyone can read.
            _storage.ReleaseSlab(block.Id);
            block.HoldsSnapshotBytes = false;
            block.HoldsModelPagedKv = false;
        }

        /// <summary>Inspect pool state. Used for telemetry and tests.</summary>
        public BlockPoolStats GetStats()
        {
            return new BlockPoolStats(
                totalBlocks: _blocks.Length,
                freeBlocks: _freeQueue.Count,
                blockSize: _blockSize);
        }
    }

    public readonly record struct BlockPoolStats(
        int totalBlocks,
        int freeBlocks,
        int blockSize);
}
