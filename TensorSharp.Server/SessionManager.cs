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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server
{
    /// <summary>
    /// Thread-safe registry of <see cref="ChatSession"/> instances. The Web UI creates
    /// one session per chat; the Ollama and OpenAI compatibility endpoints share a
    /// single built-in "default" session that survives the lifetime of the server.
    ///
    /// The manager is intentionally a thin wrapper around a concurrent dictionary so
    /// that session lookup is lock-free on the inference hot path. Session disposal
    /// always goes through <see cref="ModelService.DisposeSession"/> so that the
    /// model's KV tensors are also reset when the session being removed was active.
    /// </summary>
    public sealed class SessionManager
    {
        public const string DefaultSessionId = "__default__";

        private readonly ConcurrentDictionary<string, ChatSession> _sessions = new(StringComparer.Ordinal);
        private readonly ILogger<SessionManager> _logger;

        public SessionManager()
            : this(NullLogger<SessionManager>.Instance)
        {
        }

        public SessionManager(ILogger<SessionManager> logger)
        {
            _logger = logger ?? NullLogger<SessionManager>.Instance;
            _sessions[DefaultSessionId] = new ChatSession(DefaultSessionId);
            _logger.LogDebug(LogEventIds.SessionCreated,
                "Default session {SessionId} created", DefaultSessionId);
        }

        /// <summary>
        /// Shared session used by stateless API clients (Ollama / OpenAI compatible
        /// endpoints). Never removed from the registry so multi-turn cache reuse
        /// continues to work across requests for those clients.
        /// </summary>
        public ChatSession DefaultSession => _sessions[DefaultSessionId];

        /// <summary>Snapshot of the current session ids (for diagnostics / tests).</summary>
        public IReadOnlyList<string> SessionIds => _sessions.Keys.ToArray();

        public int SessionCount => _sessions.Count;

        /// <summary>
        /// Create a new session with a freshly-generated id. The returned session is
        /// already registered and can be looked up via <see cref="GetSession"/>.
        /// </summary>
        public ChatSession CreateSession()
        {
            while (true)
            {
                var session = new ChatSession();
                if (_sessions.TryAdd(session.Id, session))
                {
                    _logger.LogInformation(LogEventIds.SessionCreated,
                        "Created chat session {SessionId} (total sessions={SessionCount})",
                        session.Id, _sessions.Count);
                    return session;
                }
            }
        }

        /// <summary>
        /// Look up the session with the given id. Returns the default session when
        /// <paramref name="id"/> is null/empty (so endpoints can treat "no session"
        /// as "stateless client"). Returns null when the id is provided but no such
        /// session exists.
        /// </summary>
        public ChatSession GetSession(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                return DefaultSession;

            return _sessions.TryGetValue(id, out var session) ? session : null;
        }

        /// <summary>
        /// Remove the session from the registry and return it (without disposing it
        /// yet). The caller is expected to coordinate with <see cref="ModelService"/>
        /// so the model's KV tensors are reset when the active session is being
        /// removed, then call <see cref="ChatSession.Dispose"/> on the returned
        /// instance.
        ///
        /// The default session cannot be removed; this method returns null for it.
        /// </summary>
        public ChatSession TryRemove(string id)
        {
            if (string.IsNullOrWhiteSpace(id))
                return null;
            if (string.Equals(id, DefaultSessionId, StringComparison.Ordinal))
            {
                _logger.LogWarning(LogEventIds.SessionRemoved,
                    "Refusing to remove default session {SessionId}", DefaultSessionId);
                return null;
            }

            if (_sessions.TryRemove(id, out var session))
            {
                _logger.LogInformation(LogEventIds.SessionRemoved,
                    "Removed chat session {SessionId} (remaining sessions={SessionCount})",
                    id, _sessions.Count);
                return session;
            }

            _logger.LogDebug(LogEventIds.SessionRemoved,
                "TryRemove({SessionId}) found no matching session", id);
            return null;
        }
    }
}
