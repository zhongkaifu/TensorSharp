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
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Server
{
    /// <summary>
    /// An isolated chat session. Each session owns its own conversation-level
    /// <see cref="Runtime.KVCache"/> bookkeeping (token sequence + logits) plus the
    /// tracked history of assistant messages with their raw output tokens.
    ///
    /// Isolation invariants:
    ///   1. A session's <see cref="KVCache"/> and <see cref="TrackedHistory"/> are only
    ///      visible to code that holds a reference to this session. The
    ///      <see cref="ModelService"/> never mixes state between sessions.
    ///   2. Only ONE session at a time can be "active" in the model (i.e. have its
    ///      token sequence reflected in the model's per-layer K/V tensors). Switching
    ///      the active session forces a full reset of both caches, so session B
    ///      cannot observe leftover K/V data from session A.
    ///   3. <see cref="Dispose"/> clears all in-memory state held by the session so
    ///      the associated KV state is released.
    /// </summary>
    public sealed class ChatSession : IDisposable
    {
        /// <summary>Unique identifier for this session (hex, no dashes).</summary>
        public string Id { get; }

        /// <summary>Conversation KV bookkeeping for this session.</summary>
        public KVCache KVCache { get; }

        /// <summary>
        /// Tracked history - the messages whose tokens are reflected (or could be
        /// reflected) in this session's KV state. Assistant entries carry their raw
        /// output tokens so later turns can splice them back into the render.
        /// </summary>
        public List<ChatMessage> TrackedHistory { get; }

        /// <summary>Creation timestamp (UTC).</summary>
        public DateTime CreatedAt { get; }

        /// <summary>Last time this session was used for inference (UTC).</summary>
        public DateTime LastUsedAt { get; internal set; }

        /// <summary>True once <see cref="Dispose"/> has been called.</summary>
        public bool IsDisposed { get; private set; }

        public ChatSession()
            : this(Guid.NewGuid().ToString("N"))
        {
        }

        internal ChatSession(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("Session id cannot be null or empty.", nameof(id));

            Id = id;
            KVCache = new KVCache();
            TrackedHistory = new List<ChatMessage>();
            CreatedAt = DateTime.UtcNow;
            LastUsedAt = CreatedAt;
        }

        /// <summary>
        /// Drop the session's tracked history and KV cache bookkeeping. The caller
        /// is responsible for also resetting the model's per-layer K/V tensors when
        /// this session was the currently-active one (see
        /// <see cref="ModelService.DisposeSession"/>).
        /// </summary>
        public void Dispose()
        {
            if (IsDisposed)
                return;
            IsDisposed = true;
            TrackedHistory.Clear();
            KVCache.Reset();
        }
    }
}
