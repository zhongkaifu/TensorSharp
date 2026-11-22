// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class GpuLockWaitersTests
{
    [Fact]
    public void CountsWaitersPerLock()
    {
        object gpuLock = new(), other = new();
        Assert.Equal(0, GpuLockWaiters.Count(gpuLock));
        GpuLockWaiters.Enter(gpuLock);
        GpuLockWaiters.Enter(gpuLock);
        Assert.Equal(2, GpuLockWaiters.Count(gpuLock));
        Assert.Equal(0, GpuLockWaiters.Count(other));
        GpuLockWaiters.Exit(gpuLock);
        GpuLockWaiters.Exit(gpuLock);
        Assert.Equal(0, GpuLockWaiters.Count(gpuLock));
    }

    [Fact]
    public void YieldReturnsAtOnceWithoutWaiters()
    {
        var watch = Stopwatch.StartNew();
        GpuLockWaiters.YieldToWaiters(new object(), TimeSpan.FromSeconds(5));
        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(1));
    }

    [Fact]
    public void WaitingThreadTakesTheLockBeforeTheWorkersNextStep()
    {
        object gpuLock = new();
        var order = new System.Collections.Concurrent.ConcurrentQueue<string>();
        using var released = new ManualResetEventSlim();
        Task waiter;
        lock (gpuLock)
        {
            // A request thread starts waiting while the worker is inside a step.
            waiter = Task.Run(() =>
            {
                GpuLockWaiters.Enter(gpuLock);
                lock (gpuLock)
                {
                    GpuLockWaiters.Exit(gpuLock);
                    order.Enqueue("request");
                    Thread.Sleep(20);
                }
            });
            SpinWait.SpinUntil(() => GpuLockWaiters.Count(gpuLock) == 1, TimeSpan.FromSeconds(5));
            order.Enqueue("step 1");
        }
        // The worker's next step yields to the registered waiter first.
        GpuLockWaiters.YieldToWaiters(gpuLock, TimeSpan.FromSeconds(5));
        lock (gpuLock)
            order.Enqueue("step 2");
        Assert.True(waiter.Wait(TimeSpan.FromSeconds(5)));
        Assert.Equal(new[] { "step 1", "request", "step 2" }, order.ToArray());
    }

    /// <summary>
    /// EnterBetweenSteps (a spoken piece's codec stream freed past the pool, an /v1/audio request) gets the lock while a
    /// worker keeps stepping as the engine's does - yield to registered waiters, then take the lock for the step - with
    /// no gap of its own between steps. A bare lock statement could wait there for as long as the worker ran.
    /// </summary>
    [Fact]
    public async Task EnterBetweenStepsGetsInWhileTheWorkerKeepsStepping()
    {
        object gpuLock = new();
        using var stop = new CancellationTokenSource();
        long steps = 0;
        var worker = new Thread(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                GpuLockWaiters.YieldToWaiters(gpuLock, TimeSpan.FromMilliseconds(200));
                lock (gpuLock)
                {
                    Interlocked.Increment(ref steps);
                    Thread.SpinWait(2000);
                }
            }
        }) { IsBackground = true };
        worker.Start();
        try
        {
            SpinWait.SpinUntil(() => Interlocked.Read(ref steps) > 10, TimeSpan.FromSeconds(10));
            long before = Interlocked.Read(ref steps);
            Task entered = Task.Run(() =>
            {
                GpuLockWaiters.EnterBetweenSteps(gpuLock);
                try { return Interlocked.Read(ref steps); }
                finally { Monitor.Exit(gpuLock); }
            });
            Assert.True(await Task.WhenAny(entered, Task.Delay(TimeSpan.FromSeconds(30))) == entered,
                "the waiter never got the lock while the worker stepped");
            Assert.True(before > 0);
            Assert.Equal(0, GpuLockWaiters.Count(gpuLock));
            // The worker was still stepping, and still steps afterwards: the waiter took a turn between steps.
            long after = Interlocked.Read(ref steps);
            SpinWait.SpinUntil(() => Interlocked.Read(ref steps) > after + 5, TimeSpan.FromSeconds(10));
            Assert.True(Interlocked.Read(ref steps) > after);
        }
        finally
        {
            stop.Cancel();
            worker.Join();
        }
    }

    [Fact]
    public void YieldIsBoundedWhenAWaiterNeverTakesTheLock()
    {
        object gpuLock = new();
        GpuLockWaiters.Enter(gpuLock);
        try
        {
            var watch = Stopwatch.StartNew();
            GpuLockWaiters.YieldToWaiters(gpuLock, TimeSpan.FromMilliseconds(50));
            Assert.True(watch.Elapsed >= TimeSpan.FromMilliseconds(45));
            Assert.True(watch.Elapsed < TimeSpan.FromSeconds(2));
        }
        finally
        {
            GpuLockWaiters.Exit(gpuLock);
        }
    }
}
