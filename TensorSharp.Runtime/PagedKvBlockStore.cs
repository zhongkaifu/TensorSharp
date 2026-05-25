// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// In-memory block store with an explicit byte budget and LRU eviction. The
    /// store owns the underlying byte arrays; callers obtain a read-only reference
    /// via <see cref="TryGet"/>. The store is internally synchronized for the
    /// inference-loop access pattern (one writer at a time, occasional readers).
    /// </summary>
    internal sealed class PagedKvBlockStore
    {
        private readonly object _gate = new();
        private readonly Dictionary<KvBlockHash, Node> _index = new();
        private Node _head;
        private Node _tail;
        private long _residentBytes;
        private readonly long _maxBytes;
        private readonly Action<KvBlockHash, byte[]> _spillOnEvict;

        public PagedKvBlockStore(long maxBytes, Action<KvBlockHash, byte[]> spillOnEvict = null)
        {
            if (maxBytes <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxBytes));
            _maxBytes = maxBytes;
            _spillOnEvict = spillOnEvict;
        }

        public long ResidentBytes
        {
            get { lock (_gate) return _residentBytes; }
        }

        public int Count
        {
            get { lock (_gate) return _index.Count; }
        }

        public long MaxBytes => _maxBytes;

        public bool TryGet(KvBlockHash hash, [NotNullWhen(true)] out byte[] payload)
        {
            lock (_gate)
            {
                if (_index.TryGetValue(hash, out Node node))
                {
                    MoveToHead(node);
                    payload = node.Payload;
                    return true;
                }
            }

            payload = null;
            return false;
        }

        public bool Contains(KvBlockHash hash)
        {
            lock (_gate)
                return _index.ContainsKey(hash);
        }

        /// <summary>
        /// Insert <paramref name="payload"/> under <paramref name="hash"/>. The
        /// store takes ownership of the array - the caller must not mutate it
        /// afterwards. If <paramref name="hash"/> is already present the call is a
        /// no-op (existing entry is touched).
        /// </summary>
        public void Put(KvBlockHash hash, byte[] payload)
        {
            if (payload == null)
                throw new ArgumentNullException(nameof(payload));
            if (payload.Length == 0)
                return;

            // A single block larger than the cap can never fit; skip it instead of
            // thrashing the cache trying to evict everything.
            if (payload.Length > _maxBytes)
                return;

            List<(KvBlockHash Hash, byte[] Bytes)> spilled = null;
            lock (_gate)
            {
                if (_index.TryGetValue(hash, out Node existing))
                {
                    MoveToHead(existing);
                    return;
                }

                while (_residentBytes + payload.Length > _maxBytes && _tail != null)
                {
                    Node victim = _tail;
                    RemoveNode(victim);
                    _index.Remove(victim.Hash);
                    _residentBytes -= victim.Payload.Length;
                    if (_spillOnEvict != null)
                    {
                        spilled ??= new List<(KvBlockHash, byte[])>(1);
                        spilled.Add((victim.Hash, victim.Payload));
                    }
                }

                var node = new Node(hash, payload);
                _index[hash] = node;
                AddToHead(node);
                _residentBytes += payload.Length;
            }

            if (spilled != null)
            {
                foreach (var entry in spilled)
                {
                    try { _spillOnEvict(entry.Hash, entry.Bytes); }
                    catch { /* spill failures must not corrupt the in-memory tier */ }
                }
            }
        }

        public void Clear()
        {
            lock (_gate)
            {
                _index.Clear();
                _head = null;
                _tail = null;
                _residentBytes = 0;
            }
        }

        private void MoveToHead(Node node)
        {
            if (ReferenceEquals(node, _head))
                return;
            RemoveNode(node);
            AddToHead(node);
        }

        private void AddToHead(Node node)
        {
            node.Prev = null;
            node.Next = _head;
            if (_head != null)
                _head.Prev = node;
            _head = node;
            if (_tail == null)
                _tail = node;
        }

        private void RemoveNode(Node node)
        {
            if (node.Prev != null)
                node.Prev.Next = node.Next;
            else
                _head = node.Next;

            if (node.Next != null)
                node.Next.Prev = node.Prev;
            else
                _tail = node.Prev;

            node.Prev = node.Next = null;
        }

        private sealed class Node
        {
            public Node(KvBlockHash hash, byte[] payload)
            {
                Hash = hash;
                Payload = payload;
            }

            public KvBlockHash Hash { get; }
            public byte[] Payload { get; }
            public Node Prev { get; set; }
            public Node Next { get; set; }
        }
    }
}
