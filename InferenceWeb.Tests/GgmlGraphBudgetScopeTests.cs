// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

// Actual device allocation; run with TS_TEST_GGML_BACKEND=cuda and native test hooks.
public sealed class GgmlGraphBudgetScopeTests
{
    [CudaFact("TS_TEST_MODEL_DIR", "gemma-4-e4b", GgmlBackend = BackendType.GgmlCuda)]
    public void SoloGemmaDecodeReturnsGraphCreditOnModelDisposeWithoutBackendShutdown()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        var budget = new MemoryBudget([new("gpu", 32L * 1024 * 1024 * 1024)]);
        using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], includeGraphBuffers: true);
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gemma-4-e4b");
        // Two loads exercise both physical cleanup and rebuilding the same
        // process-global graph pool while keeping the backend alive.
        for (int reload = 0; reload < 2; reload++)
        {
            using (var model = ModelBase.Create(path, BackendType.GgmlCuda, 1, null, null, 1, null,
                memoryPolicy: new ModelMemoryPolicy(256, 64)))
            {
                int[] prompt = model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray();
                float[] prefill = model.Forward(prompt);
                int token = Enumerable.Range(0, prefill.Length).MaxBy(i => prefill[i]);
                float[] decode = model.Forward([token]);
                Assert.All(decode, value => Assert.True(float.IsFinite(value)));
                Assert.True(scope.ActiveAllocations > 0);
                Assert.True(budget.Snapshot().Single().Committed > 0);
            }
            // Previously the solo persistent Gemma decode graph survived this
            // boundary, retaining kind-2 credit and pointers to freed weights.
            Assert.Equal(0, scope.ActiveAllocations);
            Assert.Equal(0, budget.Snapshot().Single().Committed);
            Assert.Equal(0, budget.Snapshot().Single().Reserved);
            Assert.Null(scope.CallbackError);
        }
        scope.Dispose();
        using var reattached = new GgmlCacheBudgetScope(budget, [["gpu"]], includeGraphBuffers: true);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void GraphBuffersShareAtomicQuotaAndKeepCallbacksUntilPhysicalRelease()
    {
        using var native = new NativeFixture();
        var budget = new MemoryBudget([new("shared", 512), new("gpu", 256)]);
        using var other = budget.Reserve([new("shared", 256)]);
        other.Commit();
        using var scope = new GgmlCacheBudgetScope(budget, [["shared", "gpu"]], includeGraphBuffers: true);
        Assert.True(scope.IncludesGraphBuffers);
        IntPtr allocation = native.Allocate(0, 256);
        try
        {
            Assert.NotEqual(IntPtr.Zero, allocation);
            Assert.Equal(1, scope.ActiveAllocations);
            Assert.Equal(512, budget.Snapshot().Single(p => p.Pool == "shared").Committed);
            Assert.Equal(256, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
            Assert.Equal(IntPtr.Zero, native.Allocate(0, 1));
            Assert.Equal(1, scope.ActiveAllocations);
            Assert.Throws<InvalidOperationException>(() => scope.Dispose());
            Assert.False(budget.TrySetCapacity("gpu", 255));
            Assert.Null(scope.CallbackError);
        }
        finally { native.Free(allocation); }
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.Equal(256, budget.Snapshot().Single(p => p.Pool == "shared").Committed);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved));
        scope.Dispose(); scope.Dispose();
        using var reattached = new GgmlCacheBudgetScope(budget, [["shared", "gpu"]], includeGraphBuffers: true);
        Assert.Equal(0, reattached.ActiveAllocations);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void GraphAttachRejectsOldBuffersWhileLegacyScopeStillExcludesThem()
    {
        using var native = new NativeFixture();
        var budget = new MemoryBudget([new("gpu", 32)]);
        IntPtr before = native.Allocate(0, 64), during = IntPtr.Zero;
        try
        {
            Assert.NotEqual(IntPtr.Zero, before);
            Assert.Throws<InvalidOperationException>(() => new GgmlCacheBudgetScope(budget, [["gpu"]], true));
            using (var legacy = new GgmlCacheBudgetScope(budget, [["gpu"]]))
            {
                Assert.False(legacy.IncludesGraphBuffers);
                during = native.Allocate(0, 96);
                Assert.NotEqual(IntPtr.Zero, during); // outside the legacy cache-only contract
                Assert.Equal(0, legacy.ActiveAllocations);
                Assert.Equal(0, budget.Snapshot().Single().Committed);
            }
            Assert.Throws<InvalidOperationException>(() => new GgmlCacheBudgetScope(budget, [["gpu"]], true));
        }
        finally { native.Free(during); native.Free(before); }
        using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], true);
        Assert.Equal(IntPtr.Zero, native.Allocate(0, 33));
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.Null(scope.CallbackError);
    }

    private sealed class NativeFixture : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate IntPtr AllocateBuffer(int rank, long bytes);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void FreeBuffer(IntPtr buffer);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate int ReserveHostBuffer(long bytes);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal delegate void ClearHostBuffer();
        private readonly IntPtr _module;
        public AllocateBuffer Allocate { get; }
        public FreeBuffer Free { get; }
        public ReserveHostBuffer ReserveHost { get; }
        public ClearHostBuffer ClearHost { get; }
        public NativeFixture()
        {
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            GgmlBasicOps.ClearHostBufferCache();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            try
            {
                Allocate = Marshal.GetDelegateForFunctionPointer<AllocateBuffer>(NativeLibrary.GetExport(_module, "TSGgml_TestGraphBudgetAllocate"));
                Free = Marshal.GetDelegateForFunctionPointer<FreeBuffer>(NativeLibrary.GetExport(_module, "TSGgml_TestGraphBudgetFree"));
                ReserveHost = Marshal.GetDelegateForFunctionPointer<ReserveHostBuffer>(NativeLibrary.GetExport(_module, "TSGgml_TestExpertHostWorkspaceReserve"));
                ClearHost = Marshal.GetDelegateForFunctionPointer<ClearHostBuffer>(NativeLibrary.GetExport(_module, "TSGgml_TestExpertHostWorkspaceClear"));
            }
            catch { NativeLibrary.Free(_module); throw; }
        }
        public void Dispose() => NativeLibrary.Free(_module);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void ExpertHostArenaSharesRamCreditWithoutChargingGpuAndRejectsLateAdoption()
    {
        using var native = new NativeFixture();
        var budget = new MemoryBudget([new("ram", 8192), new("gpu", 4096)]);
        using var other = budget.Reserve([new("ram", 4096)]);
        other.Commit();
        Assert.Equal(1, native.ReserveHost(4096));
        try
        {
            Assert.Throws<InvalidOperationException>(() => new GgmlCacheBudgetScope(budget, [["gpu"]], false, ["ram"]));
            using var legacy = new GgmlCacheBudgetScope(budget, [["gpu"]]);
            Assert.False(legacy.IncludesHostBuffers);
            Assert.Equal(0, legacy.ActiveAllocations);
        }
        finally { native.ClearHost(); }
        using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], false, ["ram"]);
        Assert.True(scope.IncludesHostBuffers);
        try
        {
            Assert.Equal(1, native.ReserveHost(4096));
            Assert.Equal(1, scope.ActiveAllocations);
            Assert.Equal(1, native.ReserveHost(2048));
            Assert.Equal(1, scope.ActiveAllocations); // No second charge for reuse.
            Assert.Equal(8192, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
            Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
            Assert.Throws<InvalidOperationException>(() => scope.Dispose());
            // Growing frees the old arena before reserving the replacement.
            Assert.Equal(0, native.ReserveHost(4097));
            Assert.Equal(0, scope.ActiveAllocations);
            Assert.Equal(4096, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
            Assert.Equal(1, native.ReserveHost(4096));
            Assert.Null(scope.CallbackError);
        }
        finally { native.ClearHost(); }
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.Equal(4096, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved));
        scope.Dispose();
        using var reattached = new GgmlCacheBudgetScope(budget, [["gpu"]], false, ["ram"]);
    }
}
