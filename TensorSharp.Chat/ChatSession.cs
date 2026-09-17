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

namespace TensorSharp.Server
{
    /// <summary>
    /// An isolated chat session. Server sessions own only conversation history and
    /// assistant raw-token tracking; the inference engine owns all KV-state lifecycle.
    ///
    /// Isolation invariants:
    ///   1. A session's <see cref="Transcripts"/> are only visible to code that holds a
    ///      reference to this session, and a recorded turn is spliced into a later
    ///      request only when that request's history reproduces the turn's preceding
    ///      history AND the content the server emitted for it (see
    ///      <see cref="ConversationTranscriptStore"/>). A session shared by unrelated
    ///      clients (the stateless API session, the Web UI default session) therefore
    ///      never renders one client's generated text into another client's prompt.
    ///   2. No session is "active" in the model. Per-request KV blocks, prefix reuse, and
    ///      cleanup live inside Runtime/Scheduling; the session only names the cache
    ///      scope its requests run in (<see cref="ResolveCacheScope"/>).
    ///   3. <see cref="Dispose"/> clears all in-memory history held by the session.
    /// </summary>
    public sealed class ChatSession : IDisposable
    {
        /// <summary>Unique identifier for this session (hex, no dashes).</summary>
        public string Id { get; }

        /// <summary>
        /// The raw output tokens of the assistant turns generated in this session, one
        /// record per turn keyed by the conversation history that preceded it.
        /// </summary>
        internal ConversationTranscriptStore Transcripts { get; }

        /// <summary>Number of conversation positions with a recorded assistant turn
        /// (diagnostics and tests).</summary>
        public int TrackedTurnCount => Transcripts.Count;

        /// <summary>
        /// True for a session that stands for many unrelated conversations at once: the
        /// one the stateless OpenAI/Ollama routes use, the Web UI's default session for
        /// requests without a sessionId, and a one-shot generate session. Such a session
        /// has no conversation identity of its own, so a request's cache scope is the
        /// conversation its history proves it continues, or a fresh one.
        /// </summary>
        internal bool SharedAcrossConversations { get; }

        /// <summary>Guards transcript reads and writes against concurrent requests and a
        /// <c>newChat</c> reset.</summary>
        public object HistoryLock { get; } = new object();

        /// <summary>Incremented by every reset (<c>newChat</c>), so a new chat in the same
        /// session is a new cache scope and shares only the public prefix with the old.</summary>
        internal int ConversationEpoch { get; private set; }

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

        internal ChatSession(string id, bool sharedAcrossConversations = false)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("Session id cannot be null or empty.", nameof(id));

            Id = id;
            SharedAcrossConversations = sharedAcrossConversations;
            // A shared session holds many conversations; a Web UI chat holds one, but a
            // record for every turn of it. Both are bounded by raw tokens held.
            Transcripts = sharedAcrossConversations
                ? new ConversationTranscriptStore(maxChains: 4096, maxTokens: 8_000_000)
                : new ConversationTranscriptStore(maxChains: 2048, maxTokens: 4_000_000);
            CreatedAt = DateTime.UtcNow;
            LastUsedAt = CreatedAt;
        }

        /// <summary>
        /// The engine cache scope for a request of this session, as an opaque hash (never
        /// the session id itself, which reaches logs). A dedicated session is one
        /// conversation per epoch. A shared session uses the scope the request's history
        /// proved it continues (<paramref name="inheritedScope"/>), or a fresh one.
        /// </summary>
        internal string ResolveCacheScope(string inheritedScope)
        {
            if (SharedAcrossConversations)
                return inheritedScope ?? HashScope("lineage|" + Guid.NewGuid().ToString("N"));
            int epoch;
            string conversation;
            lock (HistoryLock)
            {
                epoch = ConversationEpoch;
                conversation = _conversationKey;
            }
            string owner = conversation != null ? "conversation|" + conversation : "session|" + Id;
            return HashScope(owner + "|" + epoch.ToString(System.Globalization.CultureInfo.InvariantCulture));
        }

        private string _conversationKey;

        /// <summary>
        /// Name the saved conversation this session serves, so every session a host opens
        /// for that conversation runs in ONE cache scope. TensorAgent opens a new session
        /// each time a chat is opened (a menu switch, a relaunch, a page re-attached after
        /// the WebView was suspended); scoped by session alone, each reopen lost the
        /// conversation's own retained state and re-prefilled everything past the system
        /// prompt. The key must identify a conversation of one user; null unbinds.
        /// </summary>
        internal void BindConversation(string conversationKey)
        {
            lock (HistoryLock)
                _conversationKey = string.IsNullOrEmpty(conversationKey) ? null : conversationKey;
        }

        private static string HashScope(string value)
            => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                System.Text.Encoding.UTF8.GetBytes(value)), 0, 8).ToLowerInvariant();

        /// <summary>Forget every recorded turn and start a new conversation epoch.
        /// Callers hold <see cref="HistoryLock"/>.</summary>
        internal void ResetConversation()
        {
            Transcripts.Clear();
            ConversationEpoch++;
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
            Transcripts.Clear();
        }
    }
}
