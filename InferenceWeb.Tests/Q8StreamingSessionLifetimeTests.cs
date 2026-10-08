// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

// Run in the CUDA lane against TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON.
// Missing native fault hooks fail explicitly rather than simulating an allocation.
public sealed class Q8StreamingSessionLifetimeTests
{
    private static readonly string[] Pools = ["shared", "gpu0"];

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void ConstructionAndCleanupFailure_RetainsOwnerAndQuotaUntilRetry()
    {
        using var native = new NativeFixture();
        long bytes = GgmlQ8StreamingSession.GetPayloadBytes(32, 4, 2);
        var budget = Budget(bytes);
        GgmlQ8StreamingSession? retained = null;
        try
        {
            native.FailNext(3); // Real cudaMalloc succeeds; construction and first cleanup fail.
            var error = Assert.Throws<GgmlQ8StreamingAllocationException>(() =>
                new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input));
            retained = error.UnreleasedSession;
            Assert.NotNull(retained);
            Assert.Equal(bytes, retained.PayloadBytes);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Reserved + pool.Committed));
            Assert.Null(budget.TryReserve([new MemoryCharge("gpu0", 1)]));
            retained.Dispose();
            AssertEmpty(budget);
            retained.Dispose();
            AssertEmpty(budget);
        }
        finally
        {
            native.FailNext(0);
            retained?.Dispose();
        }
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void FailedDispose_PreservesCommittedCreditAndSupportsRetry()
    {
        using var native = new NativeFixture();
        long bytes = GgmlQ8StreamingSession.GetPayloadBytes(32, 4, 2);
        var budget = Budget(bytes);
        using var session = new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input);
        try
        {
            native.FailNext(2);
            Assert.Throws<InvalidOperationException>(session.Dispose);
            Assert.All(budget.Snapshot(), pool =>
            {
                Assert.Equal(0, pool.Reserved);
                Assert.Equal(bytes, pool.Committed);
            });
            Assert.Null(budget.TryReserve([new MemoryCharge("shared", 1)]));
            session.Dispose();
            AssertEmpty(budget);
            using var replacement = new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input);
            Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Committed));
        }
        finally { native.FailNext(0); }
        AssertEmpty(budget);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void ExternalQuotaRefusal_HappensBeforeNativeAllocation()
    {
        using var native = new NativeFixture();
        long bytes = GgmlQ8StreamingSession.GetPayloadBytes(32, 4, 2);
        var budget = Budget(bytes);
        using var external = budget.Reserve([new MemoryCharge("shared", bytes)]);
        external.Commit();
        try
        {
            native.FailNext(1);
            Assert.Throws<MemoryPressureException>(() =>
                new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input));
            Assert.Equal(0, budget.Snapshot().Single(pool => pool.Pool == "gpu0").Committed);
            external.Dispose();
            // The pending native fault must still fire after quota is released:
            // refusal above never entered native creation or consumed the hook.
            var failure = Assert.Throws<InvalidOperationException>(() =>
                new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input));
            Assert.Contains("injected", failure.Message, StringComparison.OrdinalIgnoreCase);
            AssertEmpty(budget);
            using (var retry = new GgmlQ8StreamingSession(budget, Pools, 0, 32, 4, 2, native.Input))
                Assert.All(budget.Snapshot(), pool => Assert.Equal(bytes, pool.Committed));
            AssertEmpty(budget);
        }
        finally { native.FailNext(0); }
    }

    private static MemoryBudget Budget(long bytes)
        => new(Pools.Select(pool => new MemoryCharge(pool, bytes)));

    private static void AssertEmpty(MemoryBudget budget)
        => Assert.All(budget.Snapshot(), pool =>
        {
            Assert.Equal(0, pool.Reserved);
            Assert.Equal(0, pool.Committed);
        });

    private sealed class NativeFixture : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void InjectFailure(int flags);
        private readonly IntPtr _module;
        private readonly InjectFailure _inject;
        public IntPtr Input { get; private set; }

        public NativeFixture()
        {
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try
            {
                _inject = Marshal.GetDelegateForFunctionPointer<InjectFailure>(NativeLibrary.GetExport(
                    _module, "TSGgml_TestQ8StreamingFailNext"));
                Input = Marshal.AllocHGlobal(32 * 2 * sizeof(float));
                Marshal.Copy(Enumerable.Repeat(1.0f, 64).ToArray(), 0, Input, 64);
            }
            catch
            {
                if (Input != IntPtr.Zero) Marshal.FreeHGlobal(Input);
                NativeLibrary.Free(_module);
                throw;
            }
        }
        public void FailNext(int flags) => _inject(flags);
        public void Dispose()
        {
            _inject(0);
            Marshal.FreeHGlobal(Input);
            Input = IntPtr.Zero;
            NativeLibrary.Free(_module);
        }
    }
}
