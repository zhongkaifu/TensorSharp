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
using System.Diagnostics;
using System.Threading;

namespace TensorSharp.Models
{
    /// <summary>
    /// Turn-taking on <see cref="ModelBase.GpuComputeLock"/> between DiffusionGemma's batch scheduler,
    /// which holds the lock for a whole denoising block (up to 48 forwards; minutes on the cpu backend),
    /// and short jobs that need the model in between: a Jev structured read, the image encode of a new
    /// chat turn.
    ///
    /// <para>A job takes the lock through <see cref="Enter"/>, which registers it. The lock holder calls
    /// <see cref="Yield"/> between forwards; when a registered job is waiting it releases the lock, waits
    /// until every registered job has finished, and takes the lock back. With nothing registered
    /// <see cref="Yield"/> returns without releasing, so a chat-only workload holds the lock exactly as
    /// before. A plain <see cref="Monitor"/> exit and re-enter (what ModelBase.YieldGpuComputeLock does
    /// for the vision encoders) is not enough here: .NET monitors let the releasing thread barge back in
    /// ahead of a woken waiter, so the waiter could sit out several more forwards.</para>
    ///
    /// <para>Fairness runs both ways. A job waits for at most the forward in progress, however many chats
    /// share the block. A job that arrives while the holder is waiting to resume waits until the holder has
    /// resumed, so even a continuous stream of jobs lets the block advance one forward between them.</para>
    ///
    /// <para>A registered job may itself step out of the lock (the vision encoder's cooperative yield
    /// between its blocks). The holder does not take the lock back during that gap: it waits for the job
    /// to finish, so an image encode is not stretched by a denoising step per encoder block.</para>
    ///
    /// <para>Not re-entrant: a thread that holds the lock must not call <see cref="Enter"/>.</para>
    /// </summary>
    public sealed class DiffusionComputeTurns
    {
        // A pending job's thread is blocked in Monitor.TryEnter and is woken by the holder's exit, so it
        // normally takes the lock within microseconds. Past this long without it, someone outside this
        // protocol holds the lock (or the holder held it recursively and its exit did not free it); the
        // holder then queues on the lock like everyone else instead of waiting indefinitely.
        private static readonly TimeSpan PendingGrace = TimeSpan.FromSeconds(2);

        private readonly object _computeLock;
        private readonly object _sync = new();
        private int _pending;          // registered jobs waiting to take the lock
        private int _active;           // registered jobs that took it and have not called Exit
        private int _holdersYielding;  // holders that stepped aside and have not taken the lock back

        public DiffusionComputeTurns(object computeLock)
        {
            _computeLock = computeLock ?? throw new ArgumentNullException(nameof(computeLock));
        }

        /// <summary>True while a registered job is waiting for the lock or holds it.</summary>
        public bool HasJobs
        {
            get { lock (_sync) return _pending + _active > 0; }
        }

        /// <summary>
        /// Take the compute lock as a job the scheduler lets in between denoising forwards. Blocks until
        /// the lock is held; throws <see cref="OperationCanceledException"/> (without holding it) when
        /// <paramref name="cancellationToken"/> fires first. Pair with <see cref="Exit"/> on the same thread.
        /// </summary>
        public void Enter(CancellationToken cancellationToken = default)
        {
            lock (_sync)
            {
                // A holder that stepped aside for earlier jobs gets its turn back first.
                while (_holdersYielding > 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Monitor.Wait(_sync, 100);
                }
                _pending++;
            }
            bool entered = false;
            try
            {
                while (!(entered = Monitor.TryEnter(_computeLock, 100)))
                    cancellationToken.ThrowIfCancellationRequested();
            }
            finally
            {
                lock (_sync)
                {
                    _pending--;
                    if (entered) _active++;
                    Monitor.PulseAll(_sync);
                }
            }
        }

        /// <summary>Release the lock taken by <see cref="Enter"/>.</summary>
        public void Exit()
        {
            try { Monitor.Exit(_computeLock); }
            finally
            {
                lock (_sync)
                {
                    _active--;
                    Monitor.PulseAll(_sync);
                }
            }
        }

        /// <summary>
        /// Called by the lock holder at a point where another job may use the model. When a registered
        /// job is waiting (or is mid-job, outside the lock in a nested cooperative yield), releases the
        /// lock, waits for every registered job to finish, re-acquires it and returns true. Otherwise, or
        /// when the caller does not hold the lock, returns false at once without releasing anything.
        /// The caller must hold the lock once, not recursively: a single exit has to free it.
        /// </summary>
        public bool Yield()
        {
            if (!Monitor.IsEntered(_computeLock)) return false;
            lock (_sync)
            {
                if (_pending + _active == 0) return false;
                _holdersYielding++;
            }
            Monitor.Exit(_computeLock);
            try
            {
                lock (_sync)
                {
                    var noProgress = Stopwatch.StartNew();
                    int lastPending = _pending, lastActive = _active;
                    while (_pending + _active > 0)
                    {
                        if (_active == 0 && noProgress.Elapsed >= PendingGrace) break;
                        Monitor.Wait(_sync, 100);
                        if (_pending != lastPending || _active != lastActive)
                        {
                            lastPending = _pending;
                            lastActive = _active;
                            noProgress.Restart();
                        }
                    }
                }
            }
            finally
            {
                Monitor.Enter(_computeLock);
                lock (_sync)
                {
                    _holdersYielding--;
                    Monitor.PulseAll(_sync);
                }
            }
            return true;
        }
    }

    public sealed partial class DiffusionGemmaModel
    {
        private DiffusionComputeTurns _computeTurns;

        /// <summary>Turn-taking on <see cref="ModelBase.GpuComputeLock"/> between the batch scheduler's
        /// denoising blocks and short jobs (Jev reads, image encodes); see <see cref="DiffusionComputeTurns"/>.</summary>
        public DiffusionComputeTurns ComputeTurns
        {
            get
            {
                DiffusionComputeTurns turns = Volatile.Read(ref _computeTurns);
                if (turns != null) return turns;
                Interlocked.CompareExchange(ref _computeTurns, new DiffusionComputeTurns(GpuComputeLock), null);
                return _computeTurns;
            }
        }
    }
}
