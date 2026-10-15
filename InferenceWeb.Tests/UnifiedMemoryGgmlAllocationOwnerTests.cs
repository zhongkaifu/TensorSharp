// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryGgmlAllocationOwnerTests
{
    /// <summary>Exercise the real managed native-callback methods without native
    /// registration or buffers. No backend/device/model acceptance is implied.</summary>
    private sealed class Callbacks : IDisposable
    {
        private delegate ulong Reserve(IntPtr context, int rank, int kind, long bytes);
        private delegate int Commit(IntPtr context, ulong token);
        private delegate void Release(IntPtr context, ulong token);
        private readonly Reserve _reserve;
        private readonly Commit _commit;
        private readonly Release _release;
        private readonly GCHandle _handle;
        internal GgmlCacheBudgetScope Scope { get; }
        private static FieldInfo Field(string name) => typeof(GgmlCacheBudgetScope).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!;
        private static void Set(GgmlCacheBudgetScope scope, string name, object value) => Field(name).SetValue(scope, value);
        private static T Method<T>(string name) where T : Delegate =>
            typeof(GgmlCacheBudgetScope).GetMethod(name, BindingFlags.Static | BindingFlags.NonPublic)!.CreateDelegate<T>();

        internal Callbacks(MemoryBudget budget)
        {
            Scope = (GgmlCacheBudgetScope)RuntimeHelpers.GetUninitializedObject(typeof(GgmlCacheBudgetScope));
            Set(Scope, "_gate", new object()); Set(Scope, "_lifecycleGate", new object());
            Set(Scope, "_budget", budget); Set(Scope, "_rankPools", new[] { new[] { "device" } });
            Set(Scope, "_hostPools", Array.Empty<string>());
            Set(Scope, "_allocations", Activator.CreateInstance(Field("_allocations").FieldType)!);
            Set(Scope, "<IncludesGraphBuffers>k__BackingField", true);
            _reserve = Method<Reserve>("ReserveNative"); _commit = Method<Commit>("CommitNative"); _release = Method<Release>("ReleaseNative");
            _handle = GCHandle.Alloc(Scope);
        }
        internal ulong Allocate(long bytes, int kind = 2, bool onWorker = false)
        {
            ulong AllocateCore()
            {
                ulong token = _reserve(GCHandle.ToIntPtr(_handle), 0, kind, bytes);
                Assert.NotEqual(0UL, token); Assert.Equal(1, _commit(GCHandle.ToIntPtr(_handle), token));
                return token;
            }
            if (!onWorker) return AllocateCore();
            ulong result = 0; Exception? failure = null;
            var worker = new Thread(() => { try { result = AllocateCore(); } catch (Exception error) { failure = error; } });
            worker.Start(); Assert.True(worker.Join(TimeSpan.FromSeconds(5)));
            Assert.NotEqual(Environment.CurrentManagedThreadId, worker.ManagedThreadId);
            if (failure != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
            return result;
        }
        internal void Free(ulong token) => _release(GCHandle.ToIntPtr(_handle), token);
        public void Dispose()
        {
            try
            {
                Assert.Equal(0, Scope.ActiveAllocations); Assert.Null(Scope.CallbackError);
                Assert.Null(Field("_executionEnvelope").GetValue(Scope));
                Assert.Null(Field("_allocationOwner").GetValue(Scope));
            }
            finally { Set(Scope, "_disposed", true); _handle.Free(); }
        }
    }
    private static BudgetReservation Owner(MemoryBudget budget, long bytes = 100) => budget.Reserve([new("device", bytes)]);
    private static long Children(BudgetReservation owner) => Assert.Single(owner.LiveChildCharges).Bytes;

    [Fact]
    public void WorkerCheckpointAllocationsKeepPersistentOwnershipAfterRequestRelease()
    {
        var budget = new MemoryBudget([new("device", 256)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        ulong requestA, requestB, retained;
        using (callbacks.Scope.EnterExecution(request))
        {
            requestA = callbacks.Allocate(20, onWorker: true);
            using (callbacks.Scope.EnterAllocationOwner(checkpoint))
            {
                retained = callbacks.Allocate(64, onWorker: true);
                Assert.Equal(64, Children(checkpoint)); Assert.Equal(20, Children(request));
            }
            requestB = callbacks.Allocate(16, onWorker: true);
            Assert.Equal(36, Children(request)); Assert.Equal(64, Children(checkpoint));
        }
        request.Dispose();
        Assert.Equal(36, Assert.Single(request.OwnedCharges).Bytes);
        Assert.Equal(100, Assert.Single(checkpoint.OwnedCharges).Bytes);
        callbacks.Free(requestA); callbacks.Free(requestB);
        Assert.Equal(0, Assert.Single(request.OwnedCharges).Bytes);
        checkpoint.Dispose(); // Closing its unused allowance cannot refund the live copy.
        Assert.Equal(64, Assert.Single(budget.Snapshot()).Committed);
        callbacks.Free(retained);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Committed);
        Assert.Equal(0, Assert.Single(budget.Snapshot()).Reserved);
    }

    [Fact]
    public void OutOfOrderOwnerAndParentDisposalCannotClearTheActiveOverride()
    {
        var budget = new MemoryBudget([new("device", 300)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var outer = Owner(budget); using var inner = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        using var a = callbacks.Scope.EnterAllocationOwner(outer);
        using var b = callbacks.Scope.EnterAllocationOwner(inner);
        Assert.Throws<InvalidOperationException>(() => a.Dispose());
        Assert.Throws<InvalidOperationException>(() => execution.Dispose());
        Assert.Throws<InvalidOperationException>(() => callbacks.Scope.Dispose()); // Rejected before native detach.
        ulong innerToken = callbacks.Allocate(12); Assert.Equal(12, Children(inner));
        Assert.Equal(0, Children(outer)); Assert.Equal(0, Children(request));
        b.Dispose();
        ulong outerToken = callbacks.Allocate(13); Assert.Equal(13, Children(outer));
        a.Dispose();
        ulong requestToken = callbacks.Allocate(14); Assert.Equal(14, Children(request));
        execution.Dispose(); a.Dispose(); b.Dispose(); // Idempotent after valid restoration.
        callbacks.Free(innerToken); callbacks.Free(outerToken); callbacks.Free(requestToken);
    }

    [Fact]
    public void OverrideHandlePreparationOomPreservesThePreviousOwnerAndAllowsRetry()
    {
        var budget = new MemoryBudget([new("device", 300)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var outer = Owner(budget); using var candidate = Owner(budget);
        using var execution = callbacks.Scope.EnterExecution(request);
        using var current = callbacks.Scope.EnterAllocationOwner(outer);
        var before = budget.Snapshot().ToArray();
        callbacks.Scope.BeforeAllocationOwnerAllocation = () => throw new OutOfMemoryException("override owner injection");
        var error = Assert.Throws<OutOfMemoryException>(() => callbacks.Scope.EnterAllocationOwner(candidate));
        Assert.Equal("override owner injection", error.Message); Assert.Equal(before, budget.Snapshot().ToArray());
        ulong previous = callbacks.Allocate(11); Assert.Equal(11, Children(outer)); Assert.Equal(0, Children(candidate));
        callbacks.Scope.BeforeAllocationOwnerAllocation = null;
        using (callbacks.Scope.EnterAllocationOwner(candidate))
        {
            ulong retry = callbacks.Allocate(17); Assert.Equal(17, Children(candidate)); callbacks.Free(retry);
        }
        ulong restored = callbacks.Allocate(7); Assert.Equal(18, Children(outer));
        callbacks.Free(previous); callbacks.Free(restored);
    }

    [Fact]
    public void ForeignOwnerIsRejectedAndExecutionStillCannotNest()
    {
        var budget = new MemoryBudget([new("device", 200)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        using var foreign = Owner(new MemoryBudget([new("device", 100)]));
        Assert.Throws<ArgumentException>(() => callbacks.Scope.EnterAllocationOwner(foreign));
        using (callbacks.Scope.EnterAllocationOwner(checkpoint)) // Quiescent maintenance; previous owner is null.
        {
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            ulong maintenance = callbacks.Allocate(9); Assert.Equal(9, Children(checkpoint)); callbacks.Free(maintenance);
        }
        using (callbacks.Scope.EnterExecution(request))
        {
            Assert.Throws<ArgumentException>(() => callbacks.Scope.EnterAllocationOwner(foreign));
            Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            using (callbacks.Scope.EnterAllocationOwner(checkpoint))
                Assert.Throws<InvalidOperationException>(() => callbacks.Scope.EnterExecution(request));
            ulong restored = callbacks.Allocate(8); Assert.Equal(8, Children(request)); callbacks.Free(restored);
        }
        callbacks.Scope.BeforeExecutionOwnerAllocation = () => throw new OutOfMemoryException("execution owner injection");
        Assert.Throws<OutOfMemoryException>(() => callbacks.Scope.EnterExecution(request));
        callbacks.Scope.BeforeExecutionOwnerAllocation = null;
        using (callbacks.Scope.EnterExecution(request)) { }
    }

    [Fact]
    public void SuccessfulRestoreAllocatesNoOwnerAndSharedKindsKeepTheirDedicatedEnvelopes()
    {
        var budget = new MemoryBudget([new("device", 400)]);
        using var callbacks = new Callbacks(budget);
        using var request = Owner(budget); using var checkpoint = Owner(budget);
        using var experts = Owner(budget); using var joint = Owner(budget);
        callbacks.Scope.SetSharedExpertEnvelope(experts); callbacks.Scope.SetSharedJointEnvelope(joint);
        using (callbacks.Scope.EnterExecution(request))
        {
            for (int i = 0; i < 32; i++) { using var warm = callbacks.Scope.EnterAllocationOwner(checkpoint); }
            var owner = callbacks.Scope.EnterAllocationOwner(checkpoint);
            ulong state = callbacks.Allocate(21), expert = callbacks.Allocate(22, 4), arena = callbacks.Allocate(23, 5);
            Assert.Equal(21, Children(checkpoint)); Assert.Equal(22, Children(experts)); Assert.Equal(23, Children(joint));
            Assert.Equal(0, Children(request));
            long before = GC.GetAllocatedBytesForCurrentThread();
            owner.Dispose();
            long restoredBytes = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.Equal(0, restoredBytes);
            ulong restored = callbacks.Allocate(24); Assert.Equal(24, Children(request));
            callbacks.Free(state); callbacks.Free(expert); callbacks.Free(arena); callbacks.Free(restored);
        }
        callbacks.Scope.SetSharedExpertEnvelope(null); callbacks.Scope.SetSharedJointEnvelope(null);
    }
}
