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
    /// A captured log record produced by <see cref="FileLogger"/>. Captured here as
    /// a plain object (instead of being serialised on the producing thread) so the
    /// background writer can pay all of the formatting cost off the hot path.
    /// </summary>
    internal sealed class LogEntry
    {
        public DateTimeOffset Timestamp { get; init; }
        public LogLevel Level { get; init; }
        public string Category { get; init; }
        public EventId EventId { get; init; }
        public string Message { get; init; }
        public string Exception { get; init; }

        /// <summary>
        /// Active scope state at the time the entry was created. Each entry is a
        /// snapshot of either an <c>IEnumerable&lt;KeyValuePair&lt;string, object&gt;&gt;</c>
        /// or, when the original scope was a single value, a <c>{ "scope": value }</c>
        /// wrapper so JSON output stays uniform.
        /// </summary>
        public IReadOnlyList<KeyValuePair<string, object>> Scopes { get; init; }

        /// <summary>
        /// Structured state attached to the log call (e.g. via the
        /// <c>logger.LogInformation("...", a, b)</c> overload that bundles named
        /// arguments). Empty when the producer used a plain string.
        /// </summary>
        public IReadOnlyList<KeyValuePair<string, object>> State { get; init; }
    }
}
