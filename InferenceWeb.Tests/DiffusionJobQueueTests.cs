// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion job queue: first in, first out, one job at a time, cancellation of waiting and running jobs, progress
// in order on the compute thread, checkpoints that yield to waiting compute work. The waits below only bound a broken
// queue's hang; nothing asserts how long anything takes.
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace InferenceWeb.Tests;

public sealed class DiffusionJobQueueTests
{
    private static readonly TimeSpan Guard = TimeSpan.FromSeconds(30);

    private static void Wait(ManualResetEventSlim gate) => Assert.True(gate.Wait(Guard), "the queue did not reach the point the test waits for");

    private static T Await<T>(Task<T> task)
    {
        Assert.True(task.Wait(Guard), "the job did not finish");
        return task.Result;
    }

    private static void AwaitEnd(Task task) => Assert.True(((IAsyncResult)task).AsyncWaitHandle.WaitOne(Guard), "the job did not end");

    [Fact]
    public void JobsRunFirstInFirstOutOneAtATime()
    {
        using var queue = new DiffusionJobQueue();
        using var release = new ManualResetEventSlim();
        var order = new List<int>();
        int running = 0, overlap = 0;
        int Work(int id)
        {
            if (Interlocked.Increment(ref running) > 1) Interlocked.Increment(ref overlap);
            if (id == 0) Wait(release);
            lock (order) order.Add(id);
            Interlocked.Decrement(ref running);
            return id;
        }
        var jobs = Enumerable.Range(0, 5).Select(i => queue.Submit(_ => Work(i))).ToArray();
        Assert.Equal(new long[] { 1, 2, 3, 4, 5 }, jobs.Select(j => j.Id));
        release.Set();
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, jobs.Select(j => Await(j.Completion)));
        Assert.Equal(new[] { 0, 1, 2, 3, 4 }, order);
        Assert.Equal(0, overlap);
        Assert.All(jobs, j => Assert.Equal(DiffusionJobState.Completed, j.State));
    }

    [Fact]
    public void AWaitingJobCancelledNeverRunsAndKeepsItsPlaceOutOfTheLine()
    {
        using var queue = new DiffusionJobQueue();
        using var started = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        var ran = new List<string>();
        var first = queue.Submit(_ => { started.Set(); Wait(release); lock (ran) ran.Add("first"); return 1; });
        Wait(started);
        var second = queue.Submit(_ => { lock (ran) ran.Add("second"); return 2; });
        var third = queue.Submit(_ => { lock (ran) ran.Add("third"); return 3; });
        Assert.Equal(2, queue.Pending);
        Assert.Equal(0, queue.PositionOf(second));
        Assert.Equal(1, queue.PositionOf(third));
        Assert.Equal(-1, queue.PositionOf(first));
        Assert.Same(first, queue.Running);

        second.Cancel();
        Assert.Equal(DiffusionJobState.Canceled, second.State);
        Assert.True(second.Completion.IsCanceled);
        Assert.Equal(0, queue.PositionOf(third));
        release.Set();
        Assert.Equal(3, Await(third.Completion));
        Assert.Equal(new[] { "first", "third" }, ran);
    }

    [Fact]
    public void ARunningJobStopsAtItsNextCheckpointAndTheQueueGoesOn()
    {
        using var queue = new DiffusionJobQueue();
        using var atStep = new ManualResetEventSlim();
        using var cancelled = new ManualResetEventSlim();
        int steps = 0;
        var job = queue.Submit(context =>
        {
            for (int step = 0; step < 1000; step++)
            {
                context.Checkpoint();
                steps++;
                if (step == 2)
                {
                    atStep.Set();
                    Wait(cancelled);
                }
            }
            return steps;
        });
        Wait(atStep);
        job.Cancel();
        cancelled.Set();
        AwaitEnd(job.Completion);
        Assert.True(job.Completion.IsCanceled);
        Assert.Equal(DiffusionJobState.Canceled, job.State);
        Assert.Equal(3, steps);
        Assert.Equal("next", Await(queue.Submit(_ => "next").Completion));
    }

    [Fact]
    public void TheSubmittersTokenCancelsTheJob()
    {
        using var queue = new DiffusionJobQueue();
        var already = queue.Submit(_ => 1, cancellationToken: new CancellationToken(canceled: true));
        AwaitEnd(already.Completion);
        Assert.True(already.Completion.IsCanceled);

        using var cts = new CancellationTokenSource();
        using var running = new ManualResetEventSlim();
        var job = queue.Submit(context =>
        {
            running.Set();
            while (true) context.Checkpoint();
#pragma warning disable CS0162 // the loop ends by cancellation
            return 0;
#pragma warning restore CS0162
        }, cancellationToken: cts.Token);
        Wait(running);
        cts.Cancel();
        AwaitEnd(job.Completion);
        Assert.True(job.Completion.IsCanceled);
    }

    [Fact]
    public void ProgressReachesTheSinkInOrderOnTheComputeThread()
    {
        using var queue = new DiffusionJobQueue(new DiffusionJobQueueOptions { Name = "fake" });
        var seen = new List<(DiffusionProgress Progress, int Thread)>();
        int workThread = -1;
        var job = queue.Submit(context =>
        {
            workThread = Environment.CurrentManagedThreadId;
            context.Report(new DiffusionProgress(DiffusionPhase.Condition));
            Assert.Equal(DiffusionPhase.Condition, context.Phase);
            context.Report(new DiffusionProgress(DiffusionPhase.Denoise));
            for (int i = 1; i <= 3; i++) context.Report(new DiffusionProgress(DiffusionPhase.Denoise, i, 3, i == 2 ? "preview" : null));
            context.Report(new DiffusionProgress(DiffusionPhase.Decode));
            return true;
        }, p => seen.Add((p, Environment.CurrentManagedThreadId)));
        Assert.True(Await(job.Completion));
        Assert.Equal(new[]
        {
            new DiffusionProgress(DiffusionPhase.Condition), new DiffusionProgress(DiffusionPhase.Denoise),
            new DiffusionProgress(DiffusionPhase.Denoise, 1, 3), new DiffusionProgress(DiffusionPhase.Denoise, 2, 3, "preview"),
            new DiffusionProgress(DiffusionPhase.Denoise, 3, 3), new DiffusionProgress(DiffusionPhase.Decode),
        }, seen.Select(s => s.Progress));
        Assert.All(seen, s => Assert.Equal(workThread, s.Thread));
        Assert.Equal(new[] { 1, 2, 3 }, seen.Where(s => s.Progress.IsStep).Select(s => s.Progress.Step));
        Assert.Equal(new DiffusionProgress(DiffusionPhase.Decode), job.LastProgress);
    }

    [Fact]
    public void AFailedJobFaultsAloneAndASinkThatCancelsCancelsTheJob()
    {
        using var queue = new DiffusionJobQueue();
        var failed = queue.Submit<int>(_ => throw new InvalidOperationException("boom"));
        var cancelledBySink = queue.Submit(context =>
        {
            context.Report(new DiffusionProgress(DiffusionPhase.Denoise, 1, 6));
            return 1;
        }, _ => throw new OperationCanceledException("client gone"));
        var after = queue.Submit(_ => 7);
        Assert.Equal(7, Await(after.Completion));
        Assert.True(failed.Completion.IsFaulted);
        Assert.Equal("boom", failed.Completion.Exception!.InnerException!.Message);
        Assert.Equal(DiffusionJobState.Faulted, failed.State);
        Assert.True(cancelledBySink.Completion.IsCanceled);
    }

    [Fact]
    public void CheckpointsYieldToWaitingComputeWorkWhileTheJobHoldsTheComputeLock()
    {
        var computeLock = new object();
        int yields = 0;
        bool heldAtYield = true;
        using var queue = new DiffusionJobQueue(new DiffusionJobQueueOptions
        {
            ComputeLock = computeLock,
            YieldTurn = () =>
            {
                heldAtYield &= Monitor.IsEntered(computeLock);
                yields++;
                return false;
            },
        });
        bool heldInWork = false;
        var job = queue.Submit(context =>
        {
            heldInWork = Monitor.IsEntered(computeLock);
            for (int i = 0; i < 5; i++) context.Checkpoint();
            return 0;
        });
        Await(job.Completion);
        Assert.True(heldInWork);
        Assert.True(heldAtYield);
        Assert.Equal(5, yields);
    }

    [Fact]
    public void DisposingCancelsWaitingAndRunningJobsAndRefusesNewOnes()
    {
        var queue = new DiffusionJobQueue();
        using var running = new ManualResetEventSlim();
        var first = queue.Submit(context =>
        {
            running.Set();
            while (true) context.Checkpoint();
#pragma warning disable CS0162 // the loop ends by cancellation
            return 0;
#pragma warning restore CS0162
        });
        var waiting = queue.Submit(_ => 2);
        Wait(running);
        queue.Dispose();
        Assert.True(first.Completion.IsCanceled);
        Assert.True(waiting.Completion.IsCanceled);
        Assert.Throws<ObjectDisposedException>(() => queue.Submit(_ => 3));
        queue.Dispose();
    }
}
