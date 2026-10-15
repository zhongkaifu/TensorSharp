// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using static Fixture;

// Test request coordination, not a new production admission policy. Unpinning a
// resident page does not release its budget; it makes that page evictable. The
// caller therefore announces completed operations as well as observing credit.
sealed class PressureCompletionSignal
{
    private readonly object _gate = new();
    private TaskCompletionSource _changed = NewSignal();
    private static TaskCompletionSource NewSignal() => new(TaskCreationOptions.RunContinuationsAsynchronously);
    public Task Capture() { lock (_gate) return _changed.Task; }
    public void Pulse()
    {
        TaskCompletionSource previous;
        lock (_gate) { previous = _changed; _changed = NewSignal(); }
        previous.TrySetResult();
    }
}

static class PressureRetry
{
    // A failing loading task must not prevent a second acquire from draining.
    // Preserve the original test failure if cleanup also reports an error.
    public static async Task DrainAsync(Exception? original, Task? loading, Task<ResourceLease>? waiting)
    {
        List<Exception>? errors = null;
        void Record(Exception error)
        {
            if (!ReferenceEquals(error, original)) (errors ??= []).Add(error);
        }
        if (loading != null)
            try { await loading; }
            catch (OperationCanceledException) { }
            catch (Exception error) { Record(error); }
        if (waiting != null)
            try { (await waiting).Dispose(); }
            catch (OperationCanceledException) { }
            catch (Exception error) { Record(error); }
        if (errors != null)
        {
            if (original != null) errors.Insert(0, original);
            throw new AggregateException("Pressure test cleanup failed after draining every owned task.", errors);
        }
    }

    public static async Task<ResourceLease> AcquireAsync(Fixture fixture, ResourceKey key,
        ResourceAccess access, PressureCompletionSignal completions, CancellationToken cancellationToken,
        Action? onPressure = null)
    {
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            // Capture both before the attempt, so release/unpin during the
            // attempt cannot be lost between the pressure result and waiting.
            Task credit = fixture.Budget.ChangeSignal;
            Task completion = completions.Capture();
            try
            {
                return await fixture.Scheduler.AcquireAsync(key, fixture.Host.Location,
                    access, cancellationToken);
            }
            catch (MemoryPressureException)
            {
                onPressure?.Invoke();
                await Task.WhenAny(credit, completion).WaitAsync(cancellationToken);
            }
        }
    }
}

static partial class Cases
{
    public static async Task PressureCleanupFailure()
    {
        await using var f = new Fixture(ram: Page, chunk: 512);
        var key = f.Add("owned", true);
        using var lease = await f.Scheduler.AcquireAsync(key, f.Host.Location);
        var original = new InvalidOperationException("controlled original failure");
        var loadError = new IOException("controlled loading failure");
        var pending = new TaskCompletionSource<ResourceLease>(TaskCreationOptions.RunContinuationsAsynchronously);
        Task cleanup = PressureRetry.DrainAsync(original, Task.FromException(loadError), pending.Task);
        Check.True(!cleanup.IsCompleted, "A failed loading task must not bypass the pending acquire drain.");
        pending.SetResult(lease);
        try
        {
            await cleanup;
            throw new Exception("Expected both original and loading failures.");
        }
        catch (AggregateException error)
        {
            Check.Equal(2, error.InnerExceptions.Count);
            Check.True(ReferenceEquals(original, error.InnerExceptions[0]));
            Check.True(ReferenceEquals(loadError, error.InnerExceptions[1]));
        }
        Check.Equal(0, f.Scheduler.GetStats().ActiveLeases);
    }

    public static async Task PressureWaitUnpin()
    {
        await using var f = new Fixture(ram: Page, chunk: 512);
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var completions = new PressureCompletionSignal();
        var first = f.Add("pinned", true);
        var next = f.Add("next", true);
        using var pinned = await f.Scheduler.AcquireAsync(first, f.Host.Location);
        Task credit = f.Budget.ChangeSignal;
        var blocked = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<ResourceLease> waiting = PressureRetry.AcquireAsync(f, next, ResourceAccess.Write,
            completions, deadline.Token, () => blocked.TrySetResult());
        Exception? failure = null;
        try
        {
            await blocked.Task.WaitAsync(deadline.Token);
            Check.True(!waiting.IsCompleted, "Pressure retry must wait while the only page is pinned.");
            pinned.Dispose();
            Check.True(!credit.IsCompleted, "Unpinning must not pretend resident bytes were released.");
            completions.Pulse();
            using var admitted = await waiting;
            Check.True(f.Budget.Snapshot().All(pool => pool.Available >= 0));
        }
        catch (Exception error) { failure = error; throw; }
        finally
        {
            // On assertion failure, cancel and drain before fixture disposal.
            deadline.Cancel();
            await PressureRetry.DrainAsync(failure, null, waiting);
        }
    }

    public static async Task PressureWaitTransfer()
    {
        await using var f = new Fixture(ram: Page, chunk: 512);
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var completions = new PressureCompletionSignal();
        var source = new ControlledSource(Page);
        var first = f.Add("loading", source: source);
        var next = f.Add("next", true);
        async Task Load()
        {
            var lease = await f.Scheduler.AcquireAsync(first, f.Host.Location,
                cancellationToken: deadline.Token);
            try
            {
                var bytes = new byte[Page];
                await lease.ReadAsync(0, bytes, deadline.Token);
                Check.True(bytes.All(value => value == 37));
            }
            finally { lease.Dispose(); completions.Pulse(); }
        }
        Task loading = Load();
        Task<ResourceLease>? waiting = null;
        Exception? failure = null;
        try
        {
            await source.Started.Task.WaitAsync(deadline.Token);
            Check.Equal((long)Page + 512, f.Budget.Snapshot().Single(pool => pool.Pool == "ram").Committed);
            Check.Equal(0, f.Scheduler.GetStats().ActiveLeases); // The loader owns the unpublished page.
            var blocked = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            waiting = PressureRetry.AcquireAsync(f, next, ResourceAccess.Write, completions,
                deadline.Token, () => blocked.TrySetResult());
            await blocked.Task.WaitAsync(deadline.Token);
            Check.True(!waiting.IsCompleted, "Pressure retry must wait for the gated transfer owner.");
            source.Continue.TrySetResult();
            await loading;
            using var admitted = await waiting;
            Check.Equal(2L, f.Scheduler.GetStats().Loads);
        }
        catch (Exception error) { failure = error; throw; }
        finally
        {
            source.Continue.TrySetResult();
            deadline.Cancel();
            await PressureRetry.DrainAsync(failure, loading, waiting);
        }
    }

    public static async Task PressureWaitCancellation()
    {
        await using var f = new Fixture(ram: Page, chunk: 512);
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var first = f.Add("pinned", true);
        var next = f.Add("cancelled", true);
        using var pinned = await f.Scheduler.AcquireAsync(first, f.Host.Location);
        var before = f.Budget.Snapshot().ToArray();
        var blocked = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<ResourceLease> waiting = PressureRetry.AcquireAsync(f, next, ResourceAccess.Write,
            new PressureCompletionSignal(), deadline.Token, () => blocked.TrySetResult());
        Exception? failure = null;
        try
        {
            await blocked.Task.WaitAsync(deadline.Token);
            deadline.Cancel();
            await Check.ThrowsAsync<OperationCanceledException>(async () => (await waiting).Dispose());
            Check.True(before.SequenceEqual(f.Budget.Snapshot()), "Cancelled wait must not acquire credit.");
            Check.Equal(1, f.Scheduler.GetStats().ActiveLeases);
        }
        catch (Exception error) { failure = error; throw; }
        finally
        {
            deadline.Cancel();
            await PressureRetry.DrainAsync(failure, null, waiting);
        }
    }
}
