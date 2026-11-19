// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Runtime.Scheduling.Diffusion;

/// <summary>How a <see cref="DiffusionJobQueue"/> runs its jobs.</summary>
public sealed class DiffusionJobQueueOptions
{
    /// <summary>A name for the queue's thread and messages (the model's id).</summary>
    public string Name { get; init; } = "diffusion";

    /// <summary>The lock a job holds while it runs (the loaded model's compute lock), or null for none.</summary>
    public object? ComputeLock { get; init; }

    /// <summary>
    /// Called at a running job's checkpoints, between denoising steps, while it holds <see cref="ComputeLock"/>: gives
    /// compute work waiting for the model its turn (a DiffusionComputeTurns' Yield, which lets a Jev read or an image
    /// encode run between steps) and returns whether anything ran. Null: a checkpoint only observes cancellation.
    /// </summary>
    public Func<bool>? YieldTurn { get; init; }
}

/// <summary>Where a <see cref="DiffusionJob"/> is.</summary>
public enum DiffusionJobState { Queued, Running, Completed, Canceled, Faulted }

/// <summary>
/// One loaded diffusion model's jobs, run first in, first out, one at a time, on the queue's own compute thread. A job
/// runs phase by phase; at its checkpoints between steps it observes cancellation and yields to waiting compute turns
/// (<see cref="DiffusionJobQueueOptions.YieldTurn"/>). Cancelling a queued job removes it without running it;
/// cancelling a running job stops it at its next checkpoint (the model releases its activations and keeps its weights).
/// Progress reports reach the submitter's sink synchronously, in order, on the compute thread. A failed job faults its
/// own task and the queue goes on with the next. See docs/design/diffusion-runtime.md section 3.5.
/// </summary>
public sealed class DiffusionJobQueue : IDisposable
{
    private readonly DiffusionJobQueueOptions _options;
    private readonly object _gate = new();
    private readonly LinkedList<DiffusionJob> _queue = new();
    private DiffusionJob? _running;
    private Thread? _worker;
    private bool _disposed;
    private long _nextId;

    public DiffusionJobQueue(DiffusionJobQueueOptions? options = null) => _options = options ?? new DiffusionJobQueueOptions();

    /// <summary>Jobs waiting, not counting the running one.</summary>
    public int Pending
    {
        get { lock (_gate) return _queue.Count; }
    }

    /// <summary>The job running now, or null.</summary>
    public DiffusionJob? Running
    {
        get { lock (_gate) return _running; }
    }

    /// <summary>
    /// Queue <paramref name="work"/>. It runs after every job submitted before it, receiving a context whose token is
    /// the job's (cancelled by <see cref="DiffusionJob.Cancel"/>, by <paramref name="cancellationToken"/>, or by
    /// disposing the queue) and whose reports go to <paramref name="progress"/>. A token already cancelled returns a
    /// job that is cancelled and never runs.
    /// </summary>
    public DiffusionJob<T> Submit<T>(Func<DiffusionJobContext, T> work, Action<DiffusionProgress>? progress = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(work);
        DiffusionJob<T> job;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            job = new DiffusionJob<T>(this, ++_nextId, work, progress);
            if (cancellationToken.IsCancellationRequested)
            {
                // Never queued, so the worker cannot pick it up before the registration below would cancel it.
                job.FinishCanceled();
                return job;
            }
            _queue.AddLast(job);
            if (_worker == null)
            {
                _worker = new Thread(Loop) { IsBackground = true, Name = $"{_options.Name} diffusion queue" };
                _worker.Start();
            }
            Monitor.PulseAll(_gate);
        }
        // Outside the lock: a token cancelled since the check above runs Cancel here, which takes the job out again
        // (or, once it runs, stops it at its first checkpoint).
        if (cancellationToken.CanBeCanceled)
            job.Registration = cancellationToken.Register(static state => ((DiffusionJob)state!).Cancel(), job);
        return job;
    }

    /// <summary>Jobs ahead of <paramref name="job"/> (0: it runs next), or -1 when it is not waiting.</summary>
    public int PositionOf(DiffusionJob job)
    {
        lock (_gate)
        {
            int position = 0;
            for (LinkedListNode<DiffusionJob>? node = _queue.First; node != null; node = node.Next, position++)
                if (ReferenceEquals(node.Value, job)) return position;
            return -1;
        }
    }

    /// <summary>Stop accepting jobs, cancel the waiting ones and the running one, and wait for the running one to end.</summary>
    public void Dispose()
    {
        Thread? worker;
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            while (_queue.First is { } node)
            {
                _queue.RemoveFirst();
                node.Value.FinishCanceled();
            }
            _running?.CancelToken();
            worker = _worker;
            Monitor.PulseAll(_gate);
        }
        if (worker != null && worker != Thread.CurrentThread) worker.Join();
    }

    internal void Cancel(DiffusionJob job)
    {
        lock (_gate)
        {
            if (job.State == DiffusionJobState.Queued && _queue.Remove(job))
            {
                job.FinishCanceled();
                return;
            }
            if (job.State == DiffusionJobState.Running) job.CancelToken();
        }
    }

    internal bool YieldTurn() => _options.YieldTurn?.Invoke() == true;

    private void Loop()
    {
        while (true)
        {
            DiffusionJob job;
            lock (_gate)
            {
                while (_queue.Count == 0 && !_disposed) Monitor.Wait(_gate);
                if (_queue.First == null) return;   // disposed and drained
                job = _queue.First.Value;
                _queue.RemoveFirst();
                job.State = DiffusionJobState.Running;
                _running = job;
            }
            try
            {
                if (_options.ComputeLock is { } computeLock)
                {
                    lock (computeLock) job.Execute();
                }
                else
                {
                    job.Execute();
                }
            }
            finally
            {
                lock (_gate) _running = null;
                job.Release();
            }
        }
    }
}

/// <summary>A submitted job: its state, its last progress report and its completion.</summary>
public abstract class DiffusionJob
{
    private readonly DiffusionJobQueue _queue;
    private protected readonly CancellationTokenSource _cts = new();
    private readonly Action<DiffusionProgress>? _progress;
    private readonly object _progressGate = new();
    private volatile int _state;
    private DiffusionProgress? _last;

    private protected DiffusionJob(DiffusionJobQueue queue, long id, Action<DiffusionProgress>? progress)
    {
        _queue = queue;
        Id = id;
        _progress = progress;
    }

    /// <summary>The job's number in its queue, from 1, in submission order.</summary>
    public long Id { get; }

    public DiffusionJobState State
    {
        get => (DiffusionJobState)_state;
        internal set => _state = (int)value;
    }

    /// <summary>The job's last progress report, or null before its first.</summary>
    public DiffusionProgress? LastProgress
    {
        get { lock (_progressGate) return _last; }
    }

    /// <summary>Completes when the job has run, failed or been cancelled.</summary>
    public Task Completion => CompletionTask;

    private protected abstract Task CompletionTask { get; }

    /// <summary>Cancel the job: a queued job is removed and never runs; a running one stops at its next checkpoint.</summary>
    public void Cancel() => _queue.Cancel(this);

    internal CancellationTokenRegistration Registration { get; set; }

    internal void CancelToken()
    {
        try { _cts.Cancel(); }
        catch (ObjectDisposedException) { }
    }

    internal void Report(DiffusionProgress progress)
    {
        lock (_progressGate) _last = progress;
        _progress?.Invoke(progress);
    }

    internal bool YieldTurn() => _queue.YieldTurn();

    internal abstract void Execute();

    internal abstract void FinishCanceled();

    internal void Release()
    {
        Registration.Dispose();
        _cts.Dispose();
    }
}

/// <summary>A submitted job whose work returns a <typeparamref name="T"/>.</summary>
public sealed class DiffusionJob<T> : DiffusionJob
{
    private readonly Func<DiffusionJobContext, T> _work;
    private readonly TaskCompletionSource<T> _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);

    internal DiffusionJob(DiffusionJobQueue queue, long id, Func<DiffusionJobContext, T> work, Action<DiffusionProgress>? progress)
        : base(queue, id, progress) => _work = work;

    /// <summary>The work's result; cancelled or faulted as the job ended.</summary>
    public new Task<T> Completion => _completion.Task;

    private protected override Task CompletionTask => _completion.Task;

    internal override void Execute()
    {
        CancellationToken token = _cts.Token;
        try
        {
            token.ThrowIfCancellationRequested();
            T result = _work(new DiffusionJobContext(this, token));
            State = DiffusionJobState.Completed;
            _completion.TrySetResult(result);
        }
        catch (OperationCanceledException)
        {
            State = DiffusionJobState.Canceled;
            _completion.TrySetCanceled(token.IsCancellationRequested ? token : default);
        }
        catch (Exception error)
        {
            State = DiffusionJobState.Faulted;
            _completion.TrySetException(error);
        }
    }

    internal override void FinishCanceled()
    {
        State = DiffusionJobState.Canceled;
        CancelToken();
        _completion.TrySetCanceled(_cts.Token);
        Release();
    }
}

/// <summary>What a running job's work sees: its cancellation, its progress sink and its checkpoints.</summary>
public sealed class DiffusionJobContext
{
    private readonly DiffusionJob _job;

    internal DiffusionJobContext(DiffusionJob job, CancellationToken cancellation)
    {
        _job = job;
        Cancellation = cancellation;
    }

    /// <summary>Cancelled when the job is (its own <see cref="DiffusionJob.Cancel"/>, the submitter's token, or the queue's disposal).</summary>
    public CancellationToken Cancellation { get; }

    /// <summary>The phase the job last reported.</summary>
    public DiffusionPhase Phase { get; private set; } = DiffusionPhase.Queued;

    /// <summary>Pass a progress report to the submitter's sink (synchronously, on this thread).</summary>
    public void Report(DiffusionProgress progress)
    {
        Phase = progress.Phase;
        _job.Report(progress);
    }

    /// <summary>A point between steps where the job may stop or give the model to waiting compute work: throws
    /// <see cref="OperationCanceledException"/> when the job is cancelled, otherwise yields a turn and returns.</summary>
    public void Checkpoint()
    {
        Cancellation.ThrowIfCancellationRequested();
        _job.YieldTurn();
        Cancellation.ThrowIfCancellationRequested();
    }
}
