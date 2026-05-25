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
    /// size array of <see cref="KvBlock"/> metadata, a free queue with LRU
    /// eviction, and a content-hash index used for prefix cache hits.
    ///
    /// The pool is not thread-safe; the scheduler is the single owner.
    /// </summary>
    public sealed class BlockPool
    {
        private readonly KvBlock[] _blocks;
        private readonly FreeBlockQueue _freeQueue;
        private readonly BlockHashIndex _hashIndex;
        private readonly PagedKvStorage _storage;
        private readonly int _blockSize;

        public BlockPool(int numBlocks, int blockSize, long blockByteSize)
        {
            if (numBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numBlocks));
            if (blockSize <= 0) throw new ArgumentOutOfRangeException(nameof(blockSize));

            _blockSize = blockSize;
            _blocks = new KvBlock[numBlocks];
            _freeQueue = new FreeBlockQueue();
            _hashIndex = new BlockHashIndex();
            _storage = new PagedKvStorage(numBlocks, blockByteSize);

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

        /// <summary>Look up a block by content hash (prefix cache hit). Returns
        /// false when the hash isn't indexed.</summary>
        public bool TryFindByHash(KvBlockHash hash, out KvBlock block)
        {
            return _hashIndex.TryGet(hash, out block);
        }

        /// <summary>Bump the ref count of a block. Used when a sequence adopts a
        /// prefix-cache hit block.</summary>
        public void Touch(KvBlock block)
        {
            if (block.RefCount == 0)
                _freeQueue.Remove(block);
            block.RefCount++;
        }

        /// <summary>Allocate <paramref name="count"/> empty blocks from the free
        /// queue. Returns null when the pool is exhausted (the scheduler will
        /// then preempt). Each returned block has RefCount=1, Used=0, no hash.</summary>
        public KvBlock[] AllocateNew(int count)
        {
            if (count <= 0) return Array.Empty<KvBlock>();
            if (_freeQueue.Count < count) return null;

            var result = new KvBlock[count];
            for (int i = 0; i < count; i++)
            {
                KvBlock block = _freeQueue.Dequeue();
                EvictHashIfPresent(block);
                block.RefCount = 1;
                block.Used = 0;
                result[i] = block;
            }
            return result;
        }

        /// <summary>Decrement the ref count of each block. Blocks whose ref count
        /// hits zero are returned to the free queue (cached blocks at the back so
        /// their content is still hashable; empty blocks at the back too - the
        /// queue is LRU so newly-freed blocks are reused last).</summary>
        public void Free(IReadOnlyList<KvBlock> blocks)
        {
            if (blocks == null) return;
            for (int i = 0; i < blocks.Count; i++)
            {
                KvBlock b = blocks[i];
                if (b == null) continue;
                if (b.RefCount <= 0)
                    throw new InvalidOperationException($"Double-free of block {b.Id}");
                b.RefCount--;
                if (b.RefCount == 0)
                    _freeQueue.Enqueue(b);
            }
        }

        /// <summary>Free a single block.</summary>
        public void Free(KvBlock block)
        {
            if (block == null) return;
            if (block.RefCount <= 0)
                throw new InvalidOperationException($"Double-free of block {block.Id}");
            block.RefCount--;
            if (block.RefCount == 0)
                _freeQueue.Enqueue(block);
        }

        /// <summary>Promote a block from "being written" to "full and hashed".
        /// Called by the executor at every block boundary during prefill / decode.
        /// </summary>
        public void RegisterFullBlock(KvBlock block, KvBlockHash hash, int used)
        {
            block.Used = used;
            block.ContentHash = hash;
            _hashIndex.Register(hash, block);
        }

        /// <summary>Inspect pool state. Used for telemetry and tests.</summary>
        public BlockPoolStats GetStats()
        {
            return new BlockPoolStats(
                totalBlocks: _blocks.Length,
                freeBlocks: _freeQueue.Count,
                hashedBlocks: _hashIndex.Count,
                blockSize: _blockSize);
        }

        private void EvictHashIfPresent(KvBlock block)
        {
            if (block.ContentHash is KvBlockHash hash)
            {
                _hashIndex.Unregister(hash);
                block.ContentHash = null;
                // Drop the slab too - the new owner will rewrite it on first
                // capture, and keeping the stale bytes alive wastes memory.
                _storage.ReleaseSlab(block.Id);
            }
        }
    }

    public readonly record struct BlockPoolStats(
        int totalBlocks,
        int freeBlocks,
        int hashedBlocks,
        int blockSize);
}
