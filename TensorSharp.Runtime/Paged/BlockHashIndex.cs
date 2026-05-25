// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Generic;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Maps content hashes (<see cref="KvBlockHash"/>) to physical <see cref="KvBlock"/>s.
    /// When a sequence's prompt prefix produces a hash already in this index, the
    /// sequence can adopt the existing block (incrementing its ref count) instead
    /// of recomputing its K/V. Pure analogue of vLLM's
    /// <c>cached_block_hash_to_block</c>.
    ///
    /// At most one block per hash is stored. If a second block with the same
    /// content arrives (rare, can happen when two sequences race to the same
    /// prefix) the index keeps the first - the redundant block is left
    /// un-indexed and will simply be reclaimed when its sequence finishes.
    /// </summary>
    public sealed class BlockHashIndex
    {
        private readonly Dictionary<KvBlockHash, KvBlock> _index = new();

        public int Count => _index.Count;

        public void Register(KvBlockHash hash, KvBlock block)
        {
            _index.TryAdd(hash, block);
        }

        public bool TryGet(KvBlockHash hash, out KvBlock block)
        {
            return _index.TryGetValue(hash, out block);
        }

        public void Unregister(KvBlockHash hash)
        {
            _index.Remove(hash);
        }

        public void Clear()
        {
            _index.Clear();
        }
    }
}
