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
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// <see cref="ILogger"/> implementation that hands captured records off to a
    /// shared <see cref="FileLogQueue"/>. The actual I/O happens on a background
    /// writer thread inside <see cref="FileLoggerProvider"/>; this class is
    /// allocation-light and safe to use from any thread.
    /// </summary>
    internal sealed class FileLogger : ILogger
    {
        private readonly string _category;
        private readonly FileLogQueue _queue;
        private readonly FileLoggerScopeProvider _scopes;
        private readonly LogLevel _minLevel;

        public FileLogger(string category, FileLogQueue queue, FileLoggerScopeProvider scopes, LogLevel minLevel)
        {
            _category = category;
            _queue = queue;
            _scopes = scopes;
            _minLevel = minLevel;
        }

        public IDisposable BeginScope<TState>(TState state) where TState : notnull
            => _scopes.Push(state);

        public bool IsEnabled(LogLevel logLevel)
            => logLevel != LogLevel.None && logLevel >= _minLevel;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
        {
            if (!IsEnabled(logLevel))
                return;

            string message = formatter != null
                ? formatter(state, exception)
                : state?.ToString() ?? string.Empty;

            var entry = new LogEntry
            {
                Timestamp = DateTimeOffset.UtcNow,
                Level = logLevel,
                Category = _category,
                EventId = eventId,
                Message = message,
                Exception = exception?.ToString(),
                Scopes = _scopes.SnapshotScopes(),
                State = SnapshotState(state),
            };

            _queue.Enqueue(entry);
        }

        private static IReadOnlyList<KeyValuePair<string, object>> SnapshotState<TState>(TState state)
        {
            if (state is IReadOnlyList<KeyValuePair<string, object>> readOnly)
                return readOnly;

            if (state is IEnumerable<KeyValuePair<string, object>> pairs)
            {
                var list = new List<KeyValuePair<string, object>>();
                foreach (var kv in pairs)
                    list.Add(kv);
                return list;
            }

            return Array.Empty<KeyValuePair<string, object>>();
        }
    }
}
