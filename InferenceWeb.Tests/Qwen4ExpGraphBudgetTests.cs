// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen4ExpGraphBudgetTests : IDisposable
{
    private const long Capacity = 1024L * 1024 * 1024;
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-budget-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();

    public Qwen4ExpGraphBudgetTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "128");
        _environment.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void GraphsAndStateSnapshotsChargeSharedCreditAndReleaseOnDispose()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "model.gguf"));
        float[][]? reference = null;
        long cacheOnlyBytes = 0;
        foreach (bool graphScope in new[] { false, true, true })
        {
            GgmlBasicOps.ClearHostBufferCache();
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            var budget = new MemoryBudget([new("gpu", Capacity)]);
            using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], graphScope);
            using (var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlCuda))
            {
                float[] prefill = (float[])model.ForwardRefill([11, 17, 23, 31]).Clone();
                float[] decode = (float[])model.Forward([41]).Clone();
                long beforeSnapshot = budget.Snapshot().Single().Committed;
                model.SpecSnapshotRecurrentState();
                long afterSnapshot = budget.Snapshot().Single().Committed;
                if (graphScope)
                {
                    Assert.True(beforeSnapshot > cacheOnlyBytes);
                    Assert.True(afterSnapshot > beforeSnapshot, "The device state snapshot must own additional credit.");
                }
                else
                {
                    cacheOnlyBytes = beforeSnapshot;
                    Assert.Equal(beforeSnapshot, afterSnapshot);
                }
                float[] first = (float[])model.Forward([59]).Clone();
                model.SpecRestoreRecurrentState();
                model.SpecRewindCache(5);
                float[] replay = (float[])model.Forward([59]).Clone();
                Assert.Equal(first, replay);
                float[][] rows = [prefill, decode, first];
                if (reference == null) reference = rows;
                else for (int i = 0; i < rows.Length; ++i) Assert.Equal(reference[i], rows[i]);
                Assert.Throws<InvalidOperationException>(() => scope.Dispose());
                Assert.Null(scope.CallbackError);
            }
            Assert.Equal(0, scope.ActiveAllocations);
            Assert.All(budget.Snapshot(), p => { Assert.Equal(0, p.Reserved); Assert.Equal(0, p.Committed); });
            scope.Dispose();
        }
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void BatchedArenaRefusesExhaustedCreditAndRetriesWithoutAdvancingHolders()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "arena.gguf"));
        var budget = new MemoryBudget([new("gpu", Capacity)]);
        using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], includeGraphBuffers: true);
        using (var model = new Qwen4ExpModel(path, BackendType.GgmlCuda))
        using (var reference = new Qwen4ExpModel(path, BackendType.GgmlCuda))
        {
            string[] ids = ["first", "second"];
            int[] tokens = [37, 59], positions = [25, 28];
            var expected = new float[2][];
            for (int i = 0; i < ids.Length; i++)
            {
                Assert.True(model.BindSequenceCache(ids[i]));
                Assert.True(reference.BindSequenceCache(ids[i]));
                int[] prompt = Enumerable.Range(0, positions[i] - 1).Select(j => (11 + 31 * j + 13 * i) % 250).ToArray();
                Assert.Equal(reference.Forward(prompt), model.Forward(prompt));
                Assert.Equal(reference.Forward([197]), model.Forward([197]));
                expected[i] = (float[])reference.Forward([tokens[i]]).Clone();
            }
            model.RestorePrimaryCache(); reference.RestorePrimaryCache();
            long loaded = budget.Snapshot().Single().Committed;
            Assert.True(budget.TrySetCapacity("gpu", loaded));
            var actual = new float[2][];
            Assert.False(model.TryForwardBatchedFusedDecode(ids, tokens, positions, actual));
            Assert.Equal(0, model.ArenaBatchedDecodeSteps);
            Assert.InRange(budget.Snapshot().Single().Committed, 0, loaded);
            Assert.True(budget.TrySetCapacity("gpu", Capacity));
            Assert.True(model.TryForwardBatchedFusedDecode(ids, tokens, positions, actual), model.BatchedFusedDecodeDeclineReason);
            Assert.Equal(1, model.ArenaBatchedDecodeSteps);
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.Equal(expected[i].Length, actual[i].Length);
                Assert.True(expected[i].Zip(actual[i], (a, b) => Math.Abs((double)a - b)).Max() < .03);
            }
            Assert.Null(scope.CallbackError);
        }
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.All(budget.Snapshot(), p => { Assert.Equal(0, p.Reserved); Assert.Equal(0, p.Committed); });
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void ExhaustedSharedCreditRefusesForwardAndDisposalReturnsAllCredit()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        GgmlBasicOps.ClearHostBufferCache();
        GgmlBasicOps.ReleaseReuseComputeBuffers();
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "refusal.gguf"));
        var budget = new MemoryBudget([new("gpu", Capacity)]);
        using var scope = new GgmlCacheBudgetScope(budget, [["gpu"]], includeGraphBuffers: true);
        using (var model = (Qwen4ExpModel)ModelBase.Create(path, BackendType.GgmlCuda))
        {
            long loaded = budget.Snapshot().Single().Committed;
            Assert.True(budget.TrySetCapacity("gpu", loaded));
            Assert.ThrowsAny<Exception>(() => model.ForwardRefill([11, 17, 23, 31]));
            var usage = budget.Snapshot().Single();
            Assert.InRange(usage.Reserved + usage.Committed, 0, loaded);
            Assert.Null(scope.CallbackError);
        }
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.All(budget.Snapshot(), p => { Assert.Equal(0, p.Reserved); Assert.Equal(0, p.Committed); });
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        Directory.Delete(_directory, recursive: true);
    }
}
