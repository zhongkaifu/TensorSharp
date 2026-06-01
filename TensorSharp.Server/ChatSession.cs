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

namespace TensorSharp.Server
{
    /// <summary>
    /// An isolated chat session. Server sessions now own only conversation
    /// history and assistant raw-token tracking; the inference engine owns all
    /// KV-state lifecycle.
    ///
    /// Isolation invariants:
    ///   1. A session's <see cref="TrackedHistory"/> is only visible to code that
    ///      holds a reference to this session. The <see cref="ModelService"/>
    ///      never mixes tracked history between sessions.
    ///   2. No session is "active" in the model. Per-request KV blocks, prefix
    ///      reuse, and cleanup live inside Runtime/Scheduling.
    ///   3. <see cref="Dispose"/> clears all in-memory history held by the session.
    /// </summary>
    public sealed class ChatSession : IDisposable
    {
        /// <summary>Unique identifier for this session (hex, no dashes).</summary>
        public string Id { get; }

        /// <summary>
        /// Tracked history. Assistant entries carry raw output tokens so later
        /// turns can splice them back into the render.
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
            TrackedHistory = new List<ChatMessage>();
            CreatedAt = DateTime.UtcNow;
            LastUsedAt = CreatedAt;
        }

        /// <summary>
        /// Drop the session's tracked history. KV blocks are released by the
        /// inference engine when requests finish or are aborted.
        /// </summary>
        public void Dispose()
        {
            if (IsDisposed)
                return;
            IsDisposed = true;
            TrackedHistory.Clear();
        }
    }
}
