// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using TensorSharp.Models;

using Xunit;

namespace InferenceWeb.Tests
{
    /// <summary>
    /// Invariants of the persistent CPU worker pool. Two of these cover bugs that
    /// hung a real model run rather than failing anything: a worker that started
    /// after the first job was submitted never joined it and left the submitter
    /// spinning on a completion count that could not reach zero, and the same
    /// handshake done inside the type initializer deadlocked every worker against
    /// the cctor it was waiting on.
    /// </summary>
    public class CpuWorkerPoolTests
    {
        [Theory]
        [InlineData(1)]
        [InlineData(2)]
        [InlineData(4)]
        public void OwnedPoolHonorsThreadCountAndJoinsWorkersOnDispose(int threadCount)
        {
            using var pool = new CpuWorkerPool(threadCount);
            using var barrier = new Barrier(threadCount);
            var participants = new ConcurrentDictionary<int, Thread>();
            int submitter = Environment.CurrentManagedThreadId;
            Assert.Equal(threadCount, pool.ThreadCount);
            pool.For(threadCount, _ =>
            {
                participants.TryAdd(Environment.CurrentManagedThreadId, Thread.CurrentThread);
                Assert.True(barrier.SignalAndWait(TimeSpan.FromSeconds(5)));
            });
            Assert.Equal(threadCount, participants.Count);
            pool.Dispose();
            Assert.All(participants.Where(pair => pair.Key != submitter), pair => Assert.False(pair.Value.IsAlive));
            pool.Dispose();
        }

        [Fact]
        public void OwnedPoolNestedJobsRemainInlineAndLeavePoolUsable()
        {
            using var pool = new CpuWorkerPool(3);
            int ran = 0;
            pool.For(32, _ => pool.For(5, _ => Interlocked.Increment(ref ran)));
            Assert.Equal(160, ran);
            pool.For(7, _ => Interlocked.Increment(ref ran));
            Assert.Equal(167, ran);
        }

        [Theory]
        [InlineData(1, 1)]
        [InlineData(1, 8)]
        [InlineData(2, 1)]
        [InlineData(2, 8)]
        public void DisposalFromAnyCallbackIsRejectedAndPoolRecovers(int threadCount, int blocks)
        {
            using var pool = new CpuWorkerPool(threadCount);
            pool.For(blocks, _ => Assert.Throws<InvalidOperationException>(() => pool.Dispose()));
            int ran = 0;
            pool.For(19, _ => Interlocked.Increment(ref ran));
            Assert.Equal(19, ran);
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(7)]
        public void DisposedOwnedPoolRejectsEveryJobSize(int blocks)
        {
            using var pool = new CpuWorkerPool(2);
            pool.Dispose();
            Assert.Throws<ObjectDisposedException>(() => pool.For(blocks, _ => { }));
            Assert.Throws<ObjectDisposedException>(() => pool.For(blocks, _ => { }, CancellationToken.None));
        }

        [Theory]
        [InlineData(1)]
        [InlineData(3)]
        public void CancellationPreservesItsTokenAndPoolRecovers(int threadCount)
        {
            using var pool = new CpuWorkerPool(threadCount);
            using var cancellation = new CancellationTokenSource();
            int ran = 0;
            var error = Assert.Throws<OperationCanceledException>(() => pool.For(128, _ =>
            {
                cancellation.Cancel();
                Interlocked.Increment(ref ran);
            }, cancellation.Token));
            Assert.Equal(cancellation.Token, error.CancellationToken);
            Assert.InRange(ran, 1, threadCount);
            var alreadyCanceled = Assert.Throws<OperationCanceledException>(() =>
                pool.For(1, _ => throw new InvalidOperationException("Canceled work must not run."), cancellation.Token));
            Assert.Equal(cancellation.Token, alreadyCanceled.CancellationToken);
            int recovered = 0;
            pool.For(17, _ => Interlocked.Increment(ref recovered));
            Assert.Equal(17, recovered);
        }

        [Fact]
        public void EveryBlockRunsExactlyOnce()
        {
            const int n = 977;                       // prime, so no chunking divides it evenly
            var counts = new int[n];
            CpuWorkerPool.Shared.For(n, i => Interlocked.Increment(ref counts[i]));
            for (int i = 0; i < n; i++)
                Assert.True(counts[i] == 1, $"block {i} ran {counts[i]} times");
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(2)]
        public void DegenerateBlockCountsAreHandled(int n)
        {
            int ran = 0;
            CpuWorkerPool.Shared.For(n, _ => Interlocked.Increment(ref ran));
            Assert.Equal(n, ran);
        }

        [Fact]
        public void BackToBackJobsAllComplete()
        {
            // The steady state during decode: a few hundred small jobs in a row.
            // A missed wakeup shows up here as a hang, not as a wrong answer.
            for (int job = 0; job < 500; job++)
            {
                int ran = 0;
                CpuWorkerPool.Shared.For(64, _ => Interlocked.Increment(ref ran));
                Assert.Equal(64, ran);
            }
        }

        [Fact]
        public void JobAfterAnIdlePauseStillCompletes()
        {
            // Long enough that the workers have given up spinning and parked, which
            // is the path where a wakeup can be lost.
            CpuWorkerPool.Shared.For(8, _ => { });
            Thread.Sleep(250);

            int ran = 0;
            CpuWorkerPool.Shared.For(128, _ => Interlocked.Increment(ref ran));
            Assert.Equal(128, ran);
        }

        [Fact]
        public void NestedForRunsInlineInsteadOfDeadlocking()
        {
            int inner = 0;
            CpuWorkerPool.Shared.For(16, _ =>
                CpuWorkerPool.Shared.For(4, __ => Interlocked.Increment(ref inner)));
            Assert.Equal(64, inner);
        }

        [Fact]
        public void ExceptionPropagatesAndLeavesThePoolUsable()
        {
            var thrown = Assert.Throws<AggregateException>(() =>
                CpuWorkerPool.Shared.For(256, i =>
                {
                    if (i == 200) throw new InvalidOperationException("boom");
                }));
            Assert.Contains(thrown.InnerExceptions, e => e is InvalidOperationException);

            int ran = 0;
            CpuWorkerPool.Shared.For(32, _ => Interlocked.Increment(ref ran));
            Assert.Equal(32, ran);
        }

        [Fact]
        public void ConcurrentSubmittersAllComplete()
        {
            // The pool runs one job at a time; a second submitter has to fall back
            // to running its own blocks inline rather than block or interleave.
            const int submitters = 8, blocks = 200;
            var totals = new int[submitters];
            Parallel.For(0, submitters, s =>
                CpuWorkerPool.Shared.For(blocks, _ => Interlocked.Increment(ref totals[s])));
            for (int s = 0; s < submitters; s++)
                Assert.Equal(blocks, totals[s]);
        }

        [Fact]
        public void PoolIsWideEnoughToParallelizeButLeavesHeadroom()
        {
            // Taking every core made the spinning workers starve the rest of the CPU
            // path (which still uses the ThreadPool) and regressed prefill, so the
            // default deliberately holds some back on a big machine.
            Assert.Equal(1, CpuWorkerPool.DefaultThreadCountFor(1));
            Assert.Equal(4, CpuWorkerPool.DefaultThreadCountFor(4));
            Assert.Equal(8, CpuWorkerPool.DefaultThreadCountFor(8));
            Assert.Equal(8, CpuWorkerPool.DefaultThreadCountFor(16));
            Assert.Equal(61, CpuWorkerPool.DefaultThreadCountFor(122));
            Assert.True(CpuWorkerPool.DefaultThreadCountFor(384) < 384);
            // Apple silicon has no SMT: one worker per core.
            Assert.Equal(18, CpuWorkerPool.DefaultThreadCountFor(18, oneThreadPerCore: true));
            Assert.Equal(6, CpuWorkerPool.DefaultThreadCountFor(6, oneThreadPerCore: true));
        }

        [Fact]
        public void BlocksSeeAConsistentViewOfCapturedState()
        {
            // The body is published before the generation counter is bumped; if that
            // ordering were wrong a worker could run the PREVIOUS job's delegate.
            for (int round = 0; round < 200; round++)
            {
                int expected = round;
                var seen = new List<int>();
                var gate = new object();
                CpuWorkerPool.Shared.For(32, _ =>
                {
                    lock (gate) seen.Add(expected);
                });
                Assert.Equal(32, seen.Count);
                Assert.All(seen, v => Assert.Equal(round, v));
            }
        }
    }
}
