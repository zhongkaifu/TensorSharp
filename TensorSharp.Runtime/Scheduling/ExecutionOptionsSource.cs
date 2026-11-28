// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Threading;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// The engine's execution options and when they take effect (D04: configuration has a scope and an effective time).
    /// The options are engine-step scoped: one validated snapshot (<see cref="Current"/>, numbered by <see cref="Version"/>)
    /// is taken at each step boundary (<see cref="BeginStep"/>) and every reader of that step - the admission of the requests
    /// it schedules, the step planner, the executor and the prefix cache coordinator - reads that same snapshot. Nothing
    /// changes it mid-step.
    /// <list type="bullet">
    /// <item><see cref="Fixed"/>: a host's explicit, validated options. They change only through <see cref="Update"/>, an
    /// engine command that takes effect at the next step boundary.</item>
    /// <item><see cref="FollowEnvironment"/>: the compatibility source, used when a host gives none. The process environment
    /// is an input channel read at step boundaries only (an operator's or a test's <c>TS_*</c> change applies from the next
    /// step), never by the modules of a step on their own. An <see cref="Update"/> turns it into a fixed source.</item>
    /// </list>
    /// </summary>
    public sealed class ExecutionOptionsSource
    {
        private readonly object _gate = new();
        private bool _followsEnvironment;
        private ExecutionOptions? _pending;
        private ExecutionOptions _current;
        private long _version;

        private ExecutionOptionsSource(ExecutionOptions initial, bool followsEnvironment)
        {
            _current = initial ?? throw new ArgumentNullException(nameof(initial));
            _followsEnvironment = followsEnvironment;
            _version = 1;
        }

        /// <summary>A source of the host's explicit options, changed only by <see cref="Update"/>.</summary>
        public static ExecutionOptionsSource Fixed(ExecutionOptions options) => new(options, followsEnvironment: false);

        /// <summary>The compatibility source: the environment's <c>TS_*</c> overrides, read at each step boundary.</summary>
        public static ExecutionOptionsSource FollowEnvironment() => new(ExecutionOptions.FromEnvironment(), followsEnvironment: true);

        /// <summary>The snapshot of the current step (the last one begun).</summary>
        public ExecutionOptions Current => Volatile.Read(ref _current);

        /// <summary>The snapshot's version: it changes when a step boundary took different options.</summary>
        public long Version => Interlocked.Read(ref _version);

        /// <summary>Whether the source reads the environment at step boundaries (no host gave explicit options).</summary>
        public bool FollowsEnvironment
        {
            get { lock (_gate) return _followsEnvironment; }
        }

        /// <summary>Replaces the options from the next step boundary on (an engine command: the step running now keeps its
        /// snapshot). The source stops following the environment.</summary>
        public void Update(ExecutionOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            lock (_gate)
            {
                _pending = options;
                _followsEnvironment = false;
            }
        }

        /// <summary>A step boundary: takes a pending update (or, following the environment, re-reads it) as the snapshot every
        /// reader of the step reads. Returns that snapshot.</summary>
        public ExecutionOptions BeginStep()
        {
            ExecutionOptions next;
            lock (_gate)
            {
                if (_pending != null)
                {
                    next = _pending;
                    _pending = null;
                }
                else if (_followsEnvironment)
                    next = ExecutionOptions.FromEnvironment();
                else
                    return _current;
            }
            if (!Equals(next, _current))
            {
                Volatile.Write(ref _current, next);
                Interlocked.Increment(ref _version);
            }
            return next;
        }
    }
}
