// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Real ledger ownership at the session idle boundary, without a
/// native graph. Committed children represent already-owned arena payloads.</summary>
public sealed class UnifiedMemoryJointIdleCreditTests
{
    private const string Host = UnifiedMemoryModelSession.HostPool;
    private const string Device = UnifiedMemoryModelSession.DevicePool;

    [Fact]
    public void CancellationBeforeFirstJointStepReturnsUnusedForecastForNextSoloRequest()
    {
        var f = new Fixture();
        using var cancelled = f.Budget.Reserve([new(Device, 20)]);
        Assert.Null(f.Budget.TryReserve([new(Device, 80)]));
        cancelled.Dispose();
        f.Session.ReleaseIdleJointCredit();
        Assert.Equal(new InferenceMemoryBytes(0, 0), f.Target);
        using var solo = f.Budget.TryReserve([new(Device, 80)]);
        Assert.NotNull(solo);
        Assert.Equal(2048, f.ContextHighWater);
        Assert.All(f.Joint.OwnedCharges, c => Assert.Equal(0, c.Bytes));
        f.Joint.Dispose();
    }

    [Fact]
    public void IdleReleasePreservesPendingCommittedAndSiblingOwnership()
    {
        var f = new Fixture();
        using var pending = f.Joint.TryTake([new(Host, 10), new(Device, 15)])!;
        using var allocated = f.Joint.TryTake([new(Host, 15), new(Device, 25)])!;
        allocated.Commit();
        using var sibling = f.Budget.Reserve([new(Host, 10), new(Device, 10)]);
        sibling.Commit();
        f.Session.ReleaseIdleJointCredit();
        Assert.Equal(new InferenceMemoryBytes(25, 40), f.Target);
        Assert.All(f.Joint.Charges, c => Assert.Equal(0, c.Bytes));
        Assert.Equal(new MemoryPoolSnapshot(Device, 100, 15, 35), f.Budget.Snapshot().Single(c => c.Pool == Device));
        pending.Commit();
        allocated.Dispose(); pending.Dispose();
        // Freeing children refunds this same live joint root, not the sibling.
        Assert.Equal(40, f.Joint.Charges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(10, sibling.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        f.Session.ReleaseIdleJointCredit();
        Assert.Equal(new InferenceMemoryBytes(0, 0), f.Target);
        Assert.Equal(new MemoryPoolSnapshot(Device, 100, 0, 10), f.Budget.Snapshot().Single(c => c.Pool == Device));
        f.Joint.Dispose();
    }

    [Fact]
    public void NextPairMustReacquireReleasedCreditAndKeepsNativeContextHighWater()
    {
        var f = new Fixture();
        using var allocated = f.Joint.TryTake([new(Host, 10), new(Device, 20)])!;
        allocated.Commit();
        f.Session.ReleaseIdleJointCredit();
        Assert.Equal(new InferenceMemoryBytes(10, 20), f.Target);
        Assert.Equal(2048, f.ContextHighWater);
        using var rival = f.Budget.Reserve([new(Device, 50)]);
        var nextPair = new InferenceMemoryBytes(40, 60);
        Assert.Equal(40, nextPair.Device - f.Target.Device);
        Assert.False(f.Joint.TryResizeTotal([new(Host, nextPair.Host), new(Device, nextPair.Device)]));
        Assert.Equal(20, f.Joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        rival.Dispose();
        Assert.True(f.Joint.TryResizeTotal([new(Host, nextPair.Host), new(Device, nextPair.Device)]));
        Assert.Equal(40, f.Joint.Charges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(20, f.Joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        f.Joint.Dispose();
        Assert.Equal(20, f.Budget.Snapshot().Single(c => c.Pool == Device).Committed);
    }

    [Fact]
    public void FailedReleaseNotificationDoesNotLoseTheOwnerAndBoundaryCanRetry()
    {
        var f = new Fixture();
        using var allocated = f.Joint.TryTake([new(Device, 20)])!;
        allocated.Commit();
        _ = f.Budget.ChangeSignal;
        var failure = new IOException("notification failure");
        f.Budget.BeforeChangeNotification = () => throw failure;
        Assert.Same(failure, Assert.Throws<IOException>(f.Session.ReleaseIdleJointCredit));
        Assert.Equal(20, f.Joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        f.Budget.BeforeChangeNotification = null;
        f.Session.ReleaseIdleJointCredit();
        Assert.Equal(new InferenceMemoryBytes(0, 20), f.Target);
        Assert.Equal(2048, f.ContextHighWater);
        f.Joint.Dispose();
    }

    private sealed class Fixture
    {
        internal readonly MemoryBudget Budget = new([new(Host, 100), new(Device, 100)]);
        internal readonly BudgetReservation Joint;
        internal readonly UnifiedMemoryModelSession Session =
            (UnifiedMemoryModelSession)RuntimeHelpers.GetUninitializedObject(typeof(UnifiedMemoryModelSession));
        internal Fixture()
        {
            Joint = Budget.Reserve([new(Host, 40), new(Device, 60)]);
            Field("_jointEnvelope").SetValue(Session, Joint);
            Field("_jointTarget").SetValue(Session, new InferenceMemoryBytes(40, 60));
            Field("_jointContextHighWater").SetValue(Session, 2048);
        }
        internal InferenceMemoryBytes Target => (InferenceMemoryBytes)Field("_jointTarget").GetValue(Session)!;
        internal int ContextHighWater => (int)Field("_jointContextHighWater").GetValue(Session)!;
        private static FieldInfo Field(string name) => typeof(UnifiedMemoryModelSession).GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!;
    }
}
