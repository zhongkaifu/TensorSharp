// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class StreamingDeviceWeightCacheTests
{
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(8, false)]
    [InlineData(1, false)]
    [InlineData(8, true)]
    [InlineData(1, true)]
    public void CompleteWeightsReuseAcrossInputsAndTrimPreservesOtherOwners(int type, bool resident)
    {
        using var file = new Fixture(type);
        int[] counts = resident ? [1, 1, 1] : [19, 1, 3, 19];
        float[][] reference;
        using (var cold = new Model(file.Path, Options(Budget(), 0), resident))
            reference = counts.Select(n => cold.Forward(new int[n])).ToArray();
        var budget = Budget();
        using var other = budget.Reserve([new("gpu", 64)]); other.Commit();
        using var model = new Model(file.Path, Options(budget, 1 << 20), resident);
        for (int i = 0; i < counts.Length; i++)
        {
            Assert.Equal(reference[i], model.Forward(new int[counts[i]]));
            model.ResetKVCache(); // Weight ownership must survive normal request reset.
        }
        var usage = model.StreamingWeightUsage!.Value;
        Assert.True(usage.DeviceCacheBytes > 0 && usage.DeviceCacheHits > 0);
        Assert.Equal(usage.DeviceCacheBytes + 64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Equal(1, usage.DeviceSessionCreations);
        Assert.True(usage.WeightUploadBytes < usage.DeviceCacheHitBytes);
        if (!resident) Assert.Equal(0, usage.HostCacheBytes); // GPU source no longer duplicates RAM entries.
        long reads = usage.FileBytesRead;
        model.TrimIdleMemory();
        Assert.Equal(0, model.StreamingWeightUsage.Value.DeviceCacheBytes);
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Equal(reference[0], model.Forward(new int[counts[0]]));
        Assert.True(model.StreamingWeightUsage.Value.FileBytesRead > reads);
        model.Dispose();
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void ColdWorkspaceEvictsCacheBeforeReducingItsShapeAndDoesNotRevokeOtherOwner()
    {
        using var file = new Fixture(8);
        var budget = Budget();
        long entry = GgmlResidentWeightSession.GetPayloadBytes(0, 8, 64, 32, 8, GgmlWeightStreamingArithmetic.FullPrecision);
        using var model = new Model(file.Path, Options(budget, entry), false);
        var first = model.Forward([0]);
        Assert.Equal(entry, model.StreamingWeightUsage!.Value.DeviceCacheBytes);
        long remaining = budget.Snapshot().Single(p => p.Pool == "gpu").Available;
        using var other = budget.Reserve([new("gpu", remaining - 512)]); other.Commit();
        var second = model.Forward([1]); // Different identity/data, cache miss; only 512 free bytes.
        Assert.NotEqual(first[0], second[0]);
        Assert.Equal(0, model.StreamingWeightUsage.Value.DeviceCacheBytes);
        Assert.Equal(entry, model.StreamingWeightUsage.Value.DeviceCacheEvictedBytes);
        Assert.Equal(remaining - 512, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        model.Dispose();
        Assert.Equal(remaining - 512, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
    }

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(4)] // Replacing input in an already retained arena.
    [InlineData(8)] // Initial weight upload before publishing the entry.
    [Trait("Requires", "NativeTestHooks")]
    public void UploadAndCleanupFailureRetainQuotaUntilExplicitReset(int fault)
    {
        using var file = new Fixture(8);
        using var hooks = new Hooks();
        var budget = Budget();
        using var model = new Model(file.Path, Options(budget, 1 << 20), false);
        if (fault == 4) model.Forward([0]);
        hooks.Fail(fault | 2);
        Assert.ThrowsAny<Exception>(() => model.Forward([0]));
        Assert.True(budget.Snapshot().Single(p => p.Pool == "gpu").Committed > 0);
        Assert.Throws<InvalidOperationException>(() => model.Forward([0]));
        if (fault == 4) Assert.ThrowsAny<Exception>(model.ResetKVCache); // First physical release fails.
        Assert.True(budget.Snapshot().Single(p => p.Pool == "gpu").Committed > 0);
        model.ResetKVCache();
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        model.Forward([0]);
        model.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    [Trait("Requires", "NativeTestHooks")]
    public void FailedTrimCannotLeaveAnEntryUsableOrRefundItsQuota()
    {
        using var file = new Fixture(8);
        using var hooks = new Hooks();
        var budget = Budget();
        using var model = new Model(file.Path, Options(budget, 1 << 20), false);
        float[] expected = model.Forward([0]);
        long retained = model.StreamingWeightUsage!.Value.DeviceCacheBytes;
        hooks.Fail(2);
        Assert.ThrowsAny<Exception>(() => model.TrimIdleMemory());
        Assert.Equal(retained, model.StreamingWeightUsage.Value.DeviceCacheBytes);
        Assert.Equal(retained, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
        Assert.Throws<InvalidOperationException>(() => model.Forward([0]));
        model.ResetKVCache();
        Assert.Equal(expected, model.Forward([0]));
    }

    private static MemoryBudget Budget() => new([new("ram", 1 << 20), new("gpu", 8 << 20)]);
    private static WeightStreamingOptions Options(MemoryBudget budget, long cache) => new(budget, "ram", ["gpu"], 512, 8)
        { DeviceCacheBytes = cache, DeviceCacheReserveBytes = 0, HostCacheBytes = 64 << 10, HostCacheReserveBytes = 0 };

    private sealed class Fixture : IDisposable
    {
        internal string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"ts-device-weight-cache-{Guid.NewGuid():N}.gguf");
        internal Fixture(int type) => SyntheticGguf.Write(Path,
            [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }],
            [new SyntheticGguf.Tensor { Name = "a.weight", Dims = [64, 32], Type = (SyntheticGguf.GgmlType)type,
                Data = Enumerable.Range(0, 64 * 32).Select(i => (i % 7 - 3) / 4f).ToArray() },
             new SyntheticGguf.Tensor { Name = "b.weight", Dims = [64, 32], Type = (SyntheticGguf.GgmlType)type,
                Data = Enumerable.Range(0, 64 * 32).Select(i => (i % 5 + 1) / 4f).ToArray() }]);
        public void Dispose() => File.Delete(Path);
    }

    private sealed class Model : ModelBase
    {
        private readonly bool _resident;
        public Model(string path, WeightStreamingOptions options, bool resident) : base(path, BackendType.GgmlCuda, weightStreaming: options)
        { _resident = resident; try { LoadWeights(); } catch { Dispose(); throw; } }
        protected override GgmlWeightStreamingArithmetic StreamingWeightArithmetic
            => _resident ? GgmlWeightStreamingArithmetic.ResidentCuda : GgmlWeightStreamingArithmetic.FullPrecision;
        protected override float[] ForwardCore(int[] tokens)
        {
            using var input = CreateFloatTensor(Enumerable.Range(0, tokens.Length * 64).Select(i => (i % 11 - 5) / 8f).ToArray(), tokens.Length, 64);
            using var output = LinearForward(input, tokens[0] == 0 ? "a.weight" : "b.weight");
            return output.GetElementsAsFloat(tokens.Length * 32);
        }
        protected override void ResetKVCacheCore() { }
    }

    private sealed class Hooks : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] private delegate void Inject(int flags);
        private readonly IntPtr _module;
        private readonly Inject _inject;
        public Hooks()
        {
            GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
            _module = NativeLibrary.Load(TestGates.MappedNativeGgmlOpsPath());
            _inject = Marshal.GetDelegateForFunctionPointer<Inject>(NativeLibrary.GetExport(_module, "TSGgml_TestQ8StreamingFailNext"));
        }
        public void Fail(int flags) => _inject(flags);
        public void Dispose() { _inject(0); NativeLibrary.Free(_module); }
    }
}
