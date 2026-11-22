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
using System.Runtime.CompilerServices;
using System.Threading;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>
    /// Request threads waiting for a model's GPU compute lock (prompt media preparation),
    /// so the engine worker can let them in between steps.
    ///
    /// <para>
    /// The worker takes the lock for every step and takes it again a few microseconds
    /// after releasing it, and <see cref="Monitor"/> does not hand a released lock to the
    /// longest waiter. A request preparing its images therefore waited through several
    /// steps: with four Qwen3.8 agent lanes, a request whose screenshots were all in the
    /// embedding cache sat 2.4 s behind ~1 s prefill steps before it was even planned,
    /// while another got in at once. The worker now yields to registered waiters before
    /// its next step, so a waiting request is admitted at the next step boundary.
    /// </para>
    /// </summary>
    public static class GpuLockWaiters
    {
        private static readonly ConditionalWeakTable<object, StrongBox<int>> s_waiters = new();

        /// <summary>Registers the calling thread as waiting for <paramref name="gpuLock"/>; pair
        /// with <see cref="Exit"/> once it holds the lock (or gave up).</summary>
        public static void Enter(object gpuLock)
        {
            ArgumentNullException.ThrowIfNull(gpuLock);
            Interlocked.Increment(ref s_waiters.GetValue(gpuLock, _ => new StrongBox<int>()).Value);
        }

        public static void Exit(object gpuLock)
        {
            ArgumentNullException.ThrowIfNull(gpuLock);
            if (s_waiters.TryGetValue(gpuLock, out var box))
                Interlocked.Decrement(ref box.Value);
        }

        /// <summary>Threads currently registered as waiting for <paramref name="gpuLock"/>.</summary>
        public static int Count(object gpuLock)
            => gpuLock != null && s_waiters.TryGetValue(gpuLock, out var box) ? Volatile.Read(ref box.Value) : 0;

        /// <summary>
        /// Take <paramref name="gpuLock"/> from a thread that is not the engine worker, as a registered waiter, so the worker
        /// lets it in at its next step boundary (<see cref="YieldToWaiters"/>). A bare <c>lock</c> statement there can wait for
        /// as long as the worker keeps stepping: a released <see cref="Monitor"/> goes to whoever asks next, and the worker asks
        /// again microseconds later. Release it with <see cref="Monitor.Exit(object)"/>.
        /// </summary>
        public static void EnterBetweenSteps(object gpuLock)
        {
            Enter(gpuLock);
            try { Monitor.Enter(gpuLock); }
            finally { Exit(gpuLock); }
        }

        /// <summary>
        /// Called by the engine worker before it takes <paramref name="gpuLock"/> for a step:
        /// while another thread is waiting for the lock, give it the chance to take it, for at
        /// most <paramref name="maxWait"/>. Free when nobody waits.
        /// </summary>
        public static void YieldToWaiters(object gpuLock, TimeSpan maxWait)
        {
            if (maxWait <= TimeSpan.Zero || Count(gpuLock) == 0)
                return;
            long started = Stopwatch.GetTimestamp();
            while (Count(gpuLock) > 0 && Stopwatch.GetElapsedTime(started) < maxWait)
                Thread.Sleep(1);
        }
    }
}
