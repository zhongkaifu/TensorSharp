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
using System.Threading;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Async-local linked-list of active log scopes. Mirrors the behaviour of the
    /// built-in <c>LoggerExternalScopeProvider</c> but with deterministic
    /// normalisation into <c>KeyValuePair&lt;string, object&gt;</c> so the file
    /// writer never has to inspect arbitrary scope objects via reflection.
    ///
    /// Scopes are expected to be disposed in LIFO order (the standard
    /// <c>using (logger.BeginScope(...))</c> pattern). Disposing out of order is
    /// detected and tolerated: the disposed scope is unlinked from the chain so the
    /// remaining scopes stay visible.
    /// </summary>
    internal sealed class FileLoggerScopeProvider
    {
        private static readonly AsyncLocal<Scope> Current = new();

        public IDisposable Push<TState>(TState state)
        {
            var parent = Current.Value;
            var scope = new Scope(state, parent);
            Current.Value = scope;
            return scope;
        }

        public IReadOnlyList<KeyValuePair<string, object>> SnapshotScopes()
        {
            var head = Current.Value;
            if (head == null)
                return Array.Empty<KeyValuePair<string, object>>();

            // Walk leaf -> root, then reverse so the result reads outer -> inner,
            // matching how a reader naturally interprets nested context.
            var collected = new List<KeyValuePair<string, object>>();
            for (var node = head; node != null; node = node.Parent)
            {
                if (node.Disposed)
                    continue;
                AppendScopeProperties(node.State, collected);
            }

            collected.Reverse();
            return collected;
        }

        private static void AppendScopeProperties(object state, List<KeyValuePair<string, object>> sink)
        {
            switch (state)
            {
                case null:
                    return;
                case IEnumerable<KeyValuePair<string, object>> pairs:
                    foreach (var kv in pairs)
                        sink.Add(kv);
                    return;
                default:
                    sink.Add(new KeyValuePair<string, object>("Scope", state));
                    return;
            }
        }

        private sealed class Scope : IDisposable
        {
            public object State { get; }
            public Scope Parent { get; }
            public bool Disposed { get; private set; }

            public Scope(object state, Scope parent)
            {
                State = state;
                Parent = parent;
            }

            public void Dispose()
            {
                if (Disposed)
                    return;
                Disposed = true;

                // Fast path - LIFO disposal: simply unlink the head.
                var top = Current.Value;
                if (ReferenceEquals(top, this))
                {
                    Current.Value = Parent;
                }
                // Out-of-order disposal: leave the node marked as Disposed so
                // SnapshotScopes skips it. The chain remains intact, which is the
                // safest behaviour without mutating Parent on sibling nodes (the
                // alternative requires either a mutable Parent field or a double-
                // linked list, both of which add complexity for a rare edge case).
            }
        }
    }
}
