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
    ///      reference to this session or to another session bound to the same saved
    ///      conversation (<see cref="BindConversation(string, ChatSession)"/>), and a recorded turn is spliced into a later
    ///      request only when that request's history reproduces the turn's preceding
    ///      history AND the content the server emitted for it (see
    ///      <see cref="ConversationTranscriptStore"/>). A session shared by unrelated
    ///      clients (the stateless API session, the Web UI default session) therefore
    ///      never renders one client's generated text into another client's prompt.
    ///   2. No session is "active" in the model. Per-request KV blocks, prefix reuse, and
    ///      cleanup live inside Runtime/Scheduling; the session only names the cache
    ///      scope its requests run in (<see cref="ResolveCacheScope"/>).
    ///   3. <see cref="Dispose"/> clears all in-memory history held by the session (a
    ///      transcript store it shares with another session of the same saved conversation
    ///      stays with that session).
    /// </summary>
    public sealed class ChatSession : IDisposable
    {
        /// <summary>Unique identifier for this session (hex, no dashes).</summary>
        public string Id { get; }

        /// <summary>
        /// The raw output tokens of the assistant turns generated in this session, one
        /// record per turn keyed by the conversation history that preceded it. Sessions bound
        /// to one saved conversation share it (<see cref="BindConversation(string, ChatSession)"/>). Read and
        /// replaced under <see cref="HistoryLock"/>.
        /// </summary>
        internal ConversationTranscriptStore Transcripts { get; private set; }

        // Another session of the same conversation holds this store too: a reset or dispose
        // drops this session's reference rather than clearing records that session still needs.
        private bool _transcriptsShared;

        /// <summary>Dedicated Web UI requests wait for prior cancellation and recording.</summary>
        internal TensorSharp.Chat.WebUiTurnCoordinator WebUiTurns { get; } = new();

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
            Transcripts = NewTranscriptStore(sharedAcrossConversations);
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
            {
                var isolation = TensorSharp.Runtime.Scheduling.ExecutionOptions.FromEnvironment().PrefixCacheIsolation;
                if (isolation == TensorSharp.Runtime.Scheduling.PrefixCache.IsolationMode.Shared)
                    return HashScope("trusted-shared|" + Id);
                if (isolation == TensorSharp.Runtime.Scheduling.PrefixCache.IsolationMode.Lineage && inheritedScope != null)
                    return inheritedScope;
                return HashScope("lineage|" + Guid.NewGuid().ToString("N"));
            }
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
        internal void BindConversation(string conversationKey) => BindConversation(conversationKey, sameConversation: null);

        /// <summary>
        /// <see cref="BindConversation(string)"/>, continuing the transcripts of
        /// <paramref name="sameConversation"/>, another live session already bound to that
        /// conversation. A reopened chat sends the history its earlier session generated, and
        /// only that session's records turn its assistant messages back into the tokens the
        /// conversation's cached state holds: in a store of its own they rendered through the
        /// template, which on Qwen 3.5 re-renders a past reply without its empty thinking block,
        /// so the prompt left the cache at the first reply and the reopened chat prefilled
        /// everything after it again although it ran in the conversation's own scope.
        /// </summary>
        internal void BindConversation(string conversationKey, ChatSession sameConversation)
        {
            string key = string.IsNullOrEmpty(conversationKey) ? null : conversationKey;
            ConversationTranscriptStore shared = null;
            if (key != null && sameConversation != null && !ReferenceEquals(sameConversation, this)
                && !sameConversation.SharedAcrossConversations && !SharedAcrossConversations)
            {
                // One lock at a time: two sessions of a conversation bound together never wait on each other.
                lock (sameConversation.HistoryLock)
                {
                    if (!sameConversation.IsDisposed && string.Equals(sameConversation._conversationKey, key, StringComparison.Ordinal))
                    {
                        sameConversation._transcriptsShared = true;
                        shared = sameConversation.Transcripts;
                    }
                }
            }
            lock (HistoryLock)
            {
                _conversationKey = key;
                if (shared != null && !ReferenceEquals(shared, Transcripts))
                {
                    Transcripts = shared;
                    _transcriptsShared = true;
                }
            }
        }

        /// <summary>The saved conversation this session is bound to, or null.</summary>
        internal string ConversationKey
        {
            get { lock (HistoryLock) return _conversationKey; }
        }

        private static ConversationTranscriptStore NewTranscriptStore(bool sharedAcrossConversations)
            => sharedAcrossConversations
                ? new ConversationTranscriptStore(maxChains: 4096, maxTokens: 8_000_000)
                : new ConversationTranscriptStore(maxChains: 2048, maxTokens: 4_000_000);

        // Forget this session's records: its own store is cleared, a shared one is left to the sessions still using it.
        private void ForgetTranscripts()
        {
            if (_transcriptsShared)
            {
                Transcripts = NewTranscriptStore(SharedAcrossConversations);
                _transcriptsShared = false;
            }
            else
            {
                Transcripts.Clear();
            }
        }

        private static string HashScope(string value)
            => Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                System.Text.Encoding.UTF8.GetBytes(value)), 0, 8).ToLowerInvariant();

        /// <summary>Forget every recorded turn and start a new conversation epoch.
        /// Callers hold <see cref="HistoryLock"/>.</summary>
        internal void ResetConversation()
        {
            ForgetTranscripts();
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
            lock (HistoryLock)
                ForgetTranscripts();
        }
    }
}
