// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class MemoryRequestQueueAllocationFailureTests
{
    private static MemoryBudget Budget() => new([new("ram", 1000), new("gpu", 1000)]);
    private static readonly MemoryCharge[] Both = [new("ram", 100), new("gpu", 200)];

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void IndexFailureLeavesNoUncancellableOrphanAndSameIdCanRetry(bool afterInsert)
    {
        var budget = Budget();
        var queue = new MemoryRequestQueue(budget);
        var target = afterInsert ? MemoryRequestQueue.FaultPoint.EnqueueIndexed : MemoryRequestQueue.FaultPoint.EnqueueIndex;
        queue.FaultInjector = point => { if (point == target) throw new OutOfMemoryException("controlled index failure"); };
        Assert.Throws<OutOfMemoryException>(() => queue.Enqueue("same-id", Both));
        Assert.Equal(0, queue.WaitingCount);
        Assert.Equal(0, queue.ActiveCount);
        Assert.False(queue.CancelWaiting("same-id"));
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
        queue.FaultInjector = null;
        queue.Enqueue("same-id", Both);
        Assert.True(queue.CancelWaiting("same-id"));
        queue.Enqueue("same-id", Both);
        queue.TryAdmit()!.Dispose();
        Assert.Equal(0, queue.ActiveCount);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void AdmissionAllocationFailureKeepsHeadAndLedgerForSuccessfulRetry(int pointIndex)
    {
        var budget = Budget();
        var queue = new MemoryRequestQueue(budget);
        queue.Enqueue("first", Both);
        queue.Enqueue("second", Both);
        var before = budget.Snapshot().ToArray();
        var target = new[] { MemoryRequestQueue.FaultPoint.AdmissionOwner,
            MemoryRequestQueue.FaultPoint.AdmissionActive, MemoryRequestQueue.FaultPoint.AdmissionRegistered }[pointIndex];
        queue.FaultInjector = point => { if (point == target) throw new OutOfMemoryException("controlled admission failure"); };
        Assert.Throws<OutOfMemoryException>(() => queue.TryAdmit());
        Assert.Equal(before, budget.Snapshot());
        Assert.Equal(2, queue.WaitingCount);
        Assert.Equal(0, queue.ActiveCount);
        queue.FaultInjector = null;
        using (var first = queue.TryAdmit())
        {
            Assert.NotNull(first);
            Assert.Equal("first", first.RequestId);
        }
        using (var second = queue.TryAdmit())
        {
            Assert.NotNull(second);
            Assert.Equal("second", second.RequestId);
        }
        Assert.Equal(before, budget.Snapshot());
        queue.Enqueue("first", Both);
        Assert.True(queue.CancelWaiting("first"));
    }

    [Fact]
    public void FailedAdmissionCanBeCancelledAndFaultsStayLocalToQueueInstance()
    {
        var budget = Budget();
        var failing = new MemoryRequestQueue(budget);
        var independent = new MemoryRequestQueue(budget);
        failing.Enqueue("same-id", Both);
        failing.FaultInjector = point =>
        {
            if (point == MemoryRequestQueue.FaultPoint.AdmissionOwner) throw new OutOfMemoryException("controlled");
        };
        independent.Enqueue("other", Both);
        using (var good = independent.TryAdmit()) Assert.NotNull(good);
        Assert.Throws<OutOfMemoryException>(() => failing.TryAdmit());
        Assert.True(failing.CancelWaiting("same-id"));
        failing.FaultInjector = null;
        failing.Enqueue("same-id", Both);
        failing.TryAdmit()!.Dispose();
        Assert.Equal(0, failing.ActiveCount);
        Assert.Equal(0, independent.ActiveCount);
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
    }

    [Fact]
    public void SyntheticPostReleaseNotificationFailureStillRetiresActiveId()
    {
        var budget = Budget();
        var queue = new MemoryRequestQueue(budget);
        queue.Enqueue("same-id", Both);
        var admitted = queue.TryAdmit()!;
        Task oldSignal = budget.ChangeSignal;
        budget.BeforeChangeNotification = () => throw new OutOfMemoryException("synthetic pre-notification failure");
        Assert.Throws<OutOfMemoryException>(() => admitted.Dispose());
        Assert.Equal(0, queue.ActiveCount);
        Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(0, pool.Committed); });
        // Pulse detached this signal before the injected exception. This test
        // proves owner/ID cleanup, NOT CLR OOM recovery or successful wakeup.
        Assert.False(oldSignal.IsCompleted);
        Assert.NotSame(oldSignal, budget.ChangeSignal);
        budget.BeforeChangeNotification = null;
        admitted.Dispose(); // Already completed, including the active-ID callback.
        queue.Enqueue("same-id", Both);
        queue.TryAdmit()!.Dispose();
        Assert.Equal(0, queue.ActiveCount);
    }

    [Fact]
    public void FinishCallbackFailureIsRetryableAfterEnvelopeReleased()
    {
        var budget = Budget();
        int calls = 0;
        var admitted = new AdmittedMemoryRequest("id", budget.Reserve(Both), () =>
        {
            if (++calls == 1) throw new InvalidOperationException("controlled finish failure");
        });
        Assert.Throws<InvalidOperationException>(() => admitted.Dispose());
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved));
        admitted.Dispose();
        Assert.Equal(2, calls);
        admitted.Dispose();
        Assert.Equal(2, calls);
    }

    [Fact]
    public void AdmissionAndRollbackNotificationFailuresPreserveBothErrorsAndQueue()
    {
        var budget = Budget();
        var queue = new MemoryRequestQueue(budget);
        queue.Enqueue("id", Both);
        Task oldSignal = budget.ChangeSignal;
        var admissionError = new OutOfMemoryException("controlled admission failure");
        var notificationError = new InvalidOperationException("synthetic pre-notification failure");
        queue.FaultInjector = point =>
        {
            if (point == MemoryRequestQueue.FaultPoint.AdmissionOwner) throw admissionError;
        };
        budget.BeforeChangeNotification = () => throw notificationError;
        var error = Assert.Throws<AggregateException>(() => queue.TryAdmit());
        Assert.Same(admissionError, error.InnerExceptions[0]);
        Assert.Same(notificationError, error.InnerExceptions[1]);
        Assert.Equal(1, queue.WaitingCount);
        Assert.Equal(0, queue.ActiveCount);
        Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(0, pool.Committed); });
        Assert.False(oldSignal.IsCompleted); // No claim of successful notification recovery.
        budget.BeforeChangeNotification = null;
        queue.FaultInjector = null;
        queue.TryAdmit()!.Dispose();
        queue.Enqueue("id", Both);
        Assert.True(queue.CancelWaiting("id"));
    }

    [Fact]
    public void ReleaseNotificationAndFinishFailuresPreserveBothErrorsAndAllowFinishRetry()
    {
        var budget = Budget();
        var notificationError = new OutOfMemoryException("synthetic pre-notification failure");
        var finishError = new InvalidOperationException("controlled finish failure");
        int calls = 0;
        var admitted = new AdmittedMemoryRequest("id", budget.Reserve(Both), () =>
        {
            if (++calls == 1) throw finishError;
        });
        Task oldSignal = budget.ChangeSignal;
        budget.BeforeChangeNotification = () => throw notificationError;
        var error = Assert.Throws<AggregateException>(() => admitted.Dispose());
        Assert.Same(notificationError, error.InnerExceptions[0]);
        Assert.Same(finishError, error.InnerExceptions[1]);
        Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Reserved); Assert.Equal(0, pool.Committed); });
        Assert.False(oldSignal.IsCompleted);
        budget.BeforeChangeNotification = null;
        admitted.Dispose();
        Assert.Equal(2, calls);
        admitted.Dispose();
        Assert.Equal(2, calls);
    }
}
