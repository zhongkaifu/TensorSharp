// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

// The hook is deliberately absent from production builds. Opt in only after
// building GgmlOps with TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON; a missing export
// then fails validation rather than silently counting an unexecuted path.
public sealed class Qwen4ExpArenaCopyFaultFactAttribute : FactAttribute
{
    public Qwen4ExpArenaCopyFaultFactAttribute()
        => Skip = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_ARENA_COPY_FAULTS") != "1"
            ? "Requires TS_TEST_QWEN4EXP_ARENA_COPY_FAULTS=1 and a native test-hook build."
            : TestGates.GgmlPinSkip(BackendType.GgmlCpu);
}

/// <summary>Real GGUF loading, GDN/PLE recurrence, QSA and scheduler ownership
/// using deterministic untrained weights. These do not assess language quality.</summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpBatchedDecodeTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-batch-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(2, false)] [InlineData(3, false)] [InlineData(4, true)]
    public void ArenaParity_Cpu(int width, bool q2kxl) => RunArenaParity(BackendType.GgmlCpu, width, q2kxl);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(2, false)] [InlineData(3, false)] [InlineData(4, true)]
    public void ArenaParity_Metal(int width, bool q2kxl) => RunArenaParity(BackendType.GgmlMetal, width, q2kxl);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(2, false)] [InlineData(3, false)] [InlineData(4, true)]
    public void ArenaParity_Cuda(int width, bool q2kxl) => RunArenaParity(BackendType.GgmlCuda, width, q2kxl);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(2)] [InlineData(8)]
    public void HostOffloadedArenaParity_Metal(int offloadLayers)
        => RunArenaParity(BackendType.GgmlMetal, 3, true, offloadLayers);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(2)] [InlineData(8)]
    public void HostOffloadedArenaParity_Cuda(int offloadLayers)
        => RunArenaParity(BackendType.GgmlCuda, 3, true, offloadLayers);

    [GgmlFact(BackendType.GgmlCpu)]
    public void DenseArenaWithoutQsa_Cpu() => RunArenaParity(BackendType.GgmlCpu, 3, false, qsa: false);

    [GgmlFact(BackendType.GgmlMetal)]
    public void DenseArenaWithoutQsa_Metal() => RunArenaParity(BackendType.GgmlMetal, 3, false, qsa: false);

    [GgmlFact(BackendType.GgmlCuda)]
    public void DenseArenaWithoutQsa_Cuda() => RunArenaParity(BackendType.GgmlCuda, 3, false, qsa: false);

    [GgmlFact(BackendType.GgmlCpu)]
    public void ArenaCrossesSparseQsaThresholdAfterDenseReplay_Cpu()
        => RunSparseThresholdTransition(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlCuda)]
    public void ArenaCrossesSparseQsaThresholdAfterDenseReplay_Cuda()
        => RunSparseThresholdTransition(BackendType.GgmlCuda);

    private void RunSparseThresholdTransition(BackendType backend)
    {
        const int topK = 16;
        const int sparseWidth = topK + Qwen4ExpSyntheticModelBuilder.CompressRatio - 1;
        string path = Fixture(indexerTopK: topK);
        using var model = new Qwen4ExpModel(path, backend);
        using var reference = new Qwen4ExpModel(path, backend);
        string[] ids = ["threshold-a", "threshold-b"];
        for (int i = 0; i < ids.Length; ++i)
        {
            model.BindSequenceCache(ids[i]); reference.BindSequenceCache(ids[i]);
            Assert.Equal(reference.Forward(Prompt(i, 8)), model.Forward(Prompt(i, 8)));
            Assert.Equal(reference.Forward([197]), model.Forward([197]));
            Assert.Equal(9, model.CacheSeqLen);
            Assert.Equal(16, Field<int>(model, "_kvCacheCapacity"));
            Assert.True(Field<int>(model, "_kvCacheCapacity") <= sparseWidth);
        }
        model.RestorePrimaryCache(); reference.RestorePrimaryCache();
        long before = model.ArenaBatchedDecodeSteps;
        double worst = 0;
        for (int round = 0; round < 12; ++round)
        {
            int position = 9 + round;
            int[] tokens = [17 + round, 53 + round];
            var expected = new float[ids.Length][];
            for (int i = 0; i < ids.Length; ++i)
            {
                reference.BindSequenceCache(ids[i]);
                expected[i] = (float[])reference.Forward([tokens[i]]).Clone();
            }
            reference.RestorePrimaryCache();
            var rows = new float[ids.Length][];
            if (round == 7)
            {
                // Seven dense arena executions have filled the 16-row caches.
                // Growth flushes them and solo decode expands each cache to 32.
                // The next arena graph now contains the CUDA precision CUSTOM
                // node used by sparse QSA, reproducing the delayed failure.
                Assert.False(model.TryForwardBatchedFusedDecode(ids, tokens, [position, position], rows));
                Assert.Contains("needs cache growth", model.BatchedFusedDecodeDeclineReason);
                Assert.All(rows, Assert.Null);
                for (int i = 0; i < ids.Length; ++i)
                {
                    model.BindSequenceCache(ids[i]);
                    Assert.Equal(position, model.CacheSeqLen);
                    worst = Math.Max(worst, AssertClose(expected[i], model.Forward([tokens[i]]), "threshold growth " + i));
                    Assert.Equal(32, Field<int>(model, "_kvCacheCapacity"));
                    Assert.True(Field<int>(model, "_kvCacheCapacity") > sparseWidth);
                }
                model.RestorePrimaryCache();
            }
            else
            {
                int[] order = round % 2 == 0 ? [0, 1] : [1, 0];
                Assert.True(model.TryForwardBatchedFusedDecode(order.Select(i => ids[i]).ToArray(),
                    order.Select(i => tokens[i]).ToArray(), [position, position], rows),
                    model.BatchedFusedDecodeDeclineReason);
                for (int i = 0; i < ids.Length; ++i)
                    worst = Math.Max(worst, AssertClose(expected[order[i]], rows[i], $"threshold round {round}, row {i}"));
            }
        }
        Assert.Equal(11, model.ArenaBatchedDecodeSteps - before);
        for (int i = 0; i < ids.Length; ++i)
        {
            model.BindSequenceCache(ids[i]); reference.BindSequenceCache(ids[i]);
            Assert.Equal(21, model.CacheSeqLen);
            Assert.Equal(21, Field<int>(model, "_qsaPositionCount"));
            worst = Math.Max(worst, AssertClose(reference.Forward([113]), model.Forward([113]), "threshold final solo " + i));
        }
        output.WriteLine($"{backend}: 7 dense and 4 sparse arena steps across cache growth and request reordering; final solo state matched; worst |dlogit|={worst:G6}.");
    }

    [Qwen4ExpArenaCopyFaultFact]
    [Trait("Requires", "GgmlCpu")]
    [Trait("Requires", "NativeTestHooks")]
    public void ArenaFlushCopyFailure_FencesOnlyAffectedHolderUntilReset()
    {
        string path = Fixture();
        using var model = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var reference = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var fault = new ArenaCopyFault();
        PrepareFaultHolders(model, reference);
        var rows = new float[2][];
        Assert.True(model.TryForwardBatchedFusedDecode(["a", "b"], [17, 29], [25, 25], rows),
            model.BatchedFusedDecodeDeclineReason);
        model.BindSequenceCache("a");
        fault.Set(1);
        var error = Assert.Throws<InvalidOperationException>(() => model.SpecSnapshotRecurrentState());
        Assert.Contains("GDN flush copy failed", error.Message);
        Assert.True(Field<bool>(model, "_specStateFailed"));
        Assert.Throws<InvalidOperationException>(() => model.Forward([113]));

        // Another request's recurrence remains valid after a failed flush.
        model.BindSequenceCache("b"); reference.BindSequenceCache("b");
        reference.Forward([29]);
        AssertClose(reference.Forward([113]), model.Forward([113]), "unaffected holder after flush failure");

        model.BindSequenceCache("a"); reference.BindSequenceCache("a");
        model.ResetKVCache(); reference.ResetKVCache();
        Assert.False(Field<bool>(model, "_specStateFailed"));
        Assert.Equal(reference.Forward(Prompt(0, 24)), model.Forward(Prompt(0, 24)));
        Assert.Equal(reference.Forward([197]), model.Forward([197]));
    }

    [Qwen4ExpArenaCopyFaultFact]
    [Trait("Requires", "GgmlCpu")]
    [Trait("Requires", "NativeTestHooks")]
    public void ArenaJoinCopyFailure_DeclinesWithoutAdvancingAndSoloRetryMatches()
    {
        string path = Fixture();
        using var model = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var reference = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var fault = new ArenaCopyFault();
        PrepareFaultHolders(model, reference);
        fault.Set(1);
        var rows = new float[2][];
        Assert.False(model.TryForwardBatchedFusedDecode(["a", "b"], [17, 29], [25, 25], rows));
        Assert.Contains("GDN join copy failed", model.BatchedFusedDecodeDeclineReason);
        Assert.All(rows, Assert.Null);
        foreach (var (id, token) in new[] { ("a", 17), ("b", 29) })
        {
            model.BindSequenceCache(id); reference.BindSequenceCache(id);
            Assert.Equal(25, model.CacheSeqLen);
            Assert.False(Field<bool>(model, "_specStateFailed"));
            Assert.Equal(25, Field<int>(model, "_qsaPositionCount"));
            Assert.Equal(25, Field<int>(model, "_pleNextPos"));
            Assert.Equal(reference.Forward([token]), model.Forward([token]));
        }
        model.RestorePrimaryCache(); reference.RestorePrimaryCache();
        Assert.True(model.TryForwardBatchedFusedDecode(["b", "a"], [31, 19], [26, 26], rows),
            model.BatchedFusedDecodeDeclineReason);
        reference.BindSequenceCache("b"); AssertClose(reference.Forward([31]), rows[0], "rejoin b");
        reference.BindSequenceCache("a"); AssertClose(reference.Forward([19]), rows[1], "rejoin a");
    }

    [Qwen4ExpArenaCopyFaultFact]
    [Trait("Requires", "GgmlCpu")]
    [Trait("Requires", "NativeTestHooks")]
    public void GlobalCacheEvictionCopyFailure_RetiresGraphsAndFencesOnlyLostHolderUntilReset()
    {
        string path = Fixture();
        using var model = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var reference = new Qwen4ExpModel(path, BackendType.GgmlCpu);
        using var fault = new ArenaCopyFault();
        PrepareFaultHolders(model, reference);
        var expectedContinuations = new Dictionary<string, float[]>();
        foreach (var (id, token) in new[] { ("a", 17), ("b", 29) })
        {
            reference.BindSequenceCache(id);
            reference.Forward([token]);
            expectedContinuations[id] = (float[])reference.Forward([113]).Clone();
        }
        reference.RestorePrimaryCache();
        var rows = new float[2][];
        Assert.True(model.TryForwardBatchedFusedDecode(["a", "b"], [17, 29], [25, 25], rows),
            model.BatchedFusedDecodeDeclineReason);

        // This real process-global eviction continues freeing resident weight
        // buffers after the arena's void reset returns, including on failure.
        fault.Set(1);
        GgmlBasicOps.ClearHostBufferCache();
        string? failedId = null;
        string? unaffectedId = null;
        foreach (string id in new[] { "a", "b" })
        {
            model.BindSequenceCache(id);
            var error = Record.Exception(() => model.SpecSnapshotRecurrentState());
            if (error == null)
            {
                Assert.Null(unaffectedId);
                unaffectedId = id;
                Assert.False(Field<bool>(model, "_specStateFailed"));
            }
            else
            {
                Assert.Null(failedId);
                failedId = id;
                Assert.IsType<InvalidOperationException>(error);
                Assert.Contains("state lost during global cache reset", error.Message);
                Assert.True(Field<bool>(model, "_specStateFailed"));
                Assert.Throws<InvalidOperationException>(() => model.Forward([113]));
            }
        }
        Assert.NotNull(failedId);
        Assert.NotNull(unaffectedId);

        // A repeated eviction cannot erase the lost holder's fence. Another
        // request can rebuild against the evicted weights and keep its state.
        GgmlBasicOps.ClearHostBufferCache();
        model.BindSequenceCache(unaffectedId!);
        AssertClose(expectedContinuations[unaffectedId!], model.Forward([113]), "unaffected holder after global eviction");
        model.BindSequenceCache(failedId!);
        Assert.Throws<InvalidOperationException>(() => model.Forward([113]));

        // Even the process-global seq-state release cannot silently restore a
        // surviving holder whose authoritative arena state was already lost.
        GgmlBasicOps.Qwen4ExpReleaseAllSeqState();
        var repeatError = Assert.Throws<InvalidOperationException>(() => model.SpecSnapshotRecurrentState());
        Assert.Contains("state lost during global cache reset", repeatError.Message);

        model.ResetKVCache();
        reference.BindSequenceCache(failedId!); reference.ResetKVCache();
        Assert.False(Field<bool>(model, "_specStateFailed"));
        Assert.Equal(reference.Forward(Prompt(0, 24)), model.Forward(Prompt(0, 24)));
        Assert.Equal(reference.Forward([197]), model.Forward([197]));

        // ReleaseAllSeqState intentionally discarded this surviving holder's
        // recurrent buffers too; reset both sides before testing fresh rejoin.
        model.BindSequenceCache(unaffectedId!); reference.BindSequenceCache(unaffectedId!);
        model.ResetKVCache(); reference.ResetKVCache();
        Assert.Equal(reference.Forward(Prompt(1, 24)), model.Forward(Prompt(1, 24)));
        Assert.Equal(reference.Forward([197]), model.Forward([197]));

        // Reset released the native fence and its slot registration, so the
        // next batched graph may safely admit this holder again.
        model.RestorePrimaryCache(); reference.RestorePrimaryCache();
        Assert.True(model.TryForwardBatchedFusedDecode([failedId!, unaffectedId!], [19, 31], [25, 25], rows),
            model.BatchedFusedDecodeDeclineReason);
        reference.BindSequenceCache(failedId!); AssertClose(reference.Forward([19]), rows[0], "reset holder rejoin");
        reference.BindSequenceCache(unaffectedId!); AssertClose(reference.Forward([31]), rows[1], "surviving holder rejoin");
    }

    [Qwen4ExpArenaCopyFaultFact]
    [Trait("Requires", "GgmlCpu")]
    [Trait("Requires", "NativeTestHooks")]
    public void GlobalCacheEvictionCopyFailure_DisposeDropsActiveAndInactiveHolderFences()
    {
        string path = Fixture();
        foreach (bool checkOutFailedHolder in new[] { true, false })
        {
            var model = new Qwen4ExpModel(path, BackendType.GgmlCpu);
            bool disposed = false;
            try
            {
                using var fault = new ArenaCopyFault();
                var cachePointers = new Dictionary<string, IntPtr>();
                float[] expectedColdLogits = null!;
                foreach (var (id, seed) in new[] { ("a", 0), ("b", 1) })
                {
                    model.BindSequenceCache(id);
                    float[] logits = model.Forward(Prompt(seed, 24));
                    if (seed == 0) expectedColdLogits = (float[])logits.Clone();
                    model.Forward([197]);
                    cachePointers[id] = TensorComputePrimitives.GetStoragePointer(
                        Field<Tensor[]>(model, "_kCache").First(cache => cache != null));
                }
                model.RestorePrimaryCache();
                Assert.True(model.TryForwardBatchedFusedDecode(["a", "b"], [17, 29], [25, 25], new float[2][]),
                    model.BatchedFusedDecodeDeclineReason);
                fault.Set(1);
                GgmlBasicOps.ClearHostBufferCache();
                var failed = Assert.Single(cachePointers.Where(pair =>
                    GgmlBasicOps.Qwen4ExpArenaFlushHostPointerStatus(pair.Value) == 0));
                if (checkOutFailedHolder) model.BindSequenceCache(failed.Key);
                else model.RestorePrimaryCache(); // The failed holder stays inactive; primary is checked out.

                model.Dispose();
                disposed = true;
                // This status call only looks up the pointer; it does not
                // dereference disposed bytes. No registration may survive pool
                // reuse, whether disposal handled the holder directly or via
                // the inactive-holder dictionary.
                foreach (IntPtr pointer in cachePointers.Values)
                    Assert.Equal(1, GgmlBasicOps.Qwen4ExpArenaFlushHostPointerStatus(pointer));
                using var reloaded = new Qwen4ExpModel(path, BackendType.GgmlCpu);
                Assert.Equal(expectedColdLogits, reloaded.Forward(Prompt(0, 24)));
            }
            finally { if (!disposed) model.Dispose(); }
        }
    }

    private static void PrepareFaultHolders(Qwen4ExpModel model, Qwen4ExpModel reference)
    {
        foreach (var (id, seed) in new[] { ("a", 0), ("b", 1) })
        {
            model.BindSequenceCache(id); reference.BindSequenceCache(id);
            Assert.Equal(reference.Forward(Prompt(seed, 24)), model.Forward(Prompt(seed, 24)));
            Assert.Equal(reference.Forward([197]), model.Forward([197]));
        }
        model.RestorePrimaryCache(); reference.RestorePrimaryCache();
    }

    private sealed class ArenaCopyFault : IDisposable
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int SetCopyFault(int nth);
        private readonly IntPtr _module = NativeLibrary.Load(Qwen4ExpExpertCacheScenario.MappedNativePath());
        private readonly SetCopyFault _set;
        public ArenaCopyFault()
        {
            try
            {
                _set = Marshal.GetDelegateForFunctionPointer<SetCopyFault>(
                    NativeLibrary.GetExport(_module, "TSGgml_Qwen4ExpArenaTestCopyFault"));
            }
            catch { NativeLibrary.Free(_module); throw; }
        }
        public void Set(int nth) => Assert.Equal(1, _set(nth));
        public void Dispose() { _set(0); NativeLibrary.Free(_module); }
    }

    private string Fixture(bool q2kxl = false, bool qsa = true, int indexerTopK = 16)
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "128");
        _environment.Set("TS_KV_INITIAL_TOKENS", "8");
        _environment.Set("TS_PER_SEQ_FUSED", "1");
        _environment.Set("TS_Q4E_DISABLE_ARENA_DECODE", null);
        _environment.ClearSpeculationVars();
        MoeCpuOffloadConfig.Reset();
        return Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "fixture.gguf"),
            indexerTopK: indexerTopK, q2kxlExperts: q2kxl, qsa: qsa);
    }

    private void RunArenaParity(BackendType backend, int width, bool q2kxl, int offloadLayers = 0, bool qsa = true)
    {
        string path = Fixture(q2kxl, qsa);
        if (offloadLayers > 0) MoeCpuOffloadConfig.SetLayers(offloadLayers);
        using var model = new Qwen4ExpModel(path, backend);
        using var reference = new Qwen4ExpModel(path, backend);
        var ids = Enumerable.Range(0, width).Select(i => "request-" + i).ToArray();
        var positions = new int[width];
        for (int i = 0; i < width; ++i)
        {
            int[] prompt = Prompt(i, 24 + 3 * i);
            // The first request starts solo, then a second arrival adopts its
            // live primary state, exactly as the engine transitions to batching.
            if (i != 0) Assert.True(model.BindSequenceCache(ids[i]));
            Assert.True(reference.BindSequenceCache(ids[i]));
            if (i == 0)
            {
                int[] axes = MediaAxes(prompt.Length);
                model.SetMRoPEPositions(axes); reference.SetMRoPEPositions((int[])axes.Clone());
            }
            Assert.Equal(reference.Forward(prompt), model.Forward(prompt));
            if (i == 0) model.AdoptPrimaryCacheToFused(ids[i]);
            // A prefill can grow to exactly the prompt length. The next solo
            // step grows it again, giving the arena room without hiding growth.
            Assert.Equal(reference.Forward([197]), model.Forward([197]));
            positions[i] = prompt.Length + 1;
        }
        model.RestorePrimaryCache(); reference.RestorePrimaryCache();
        double worst = 0;
        long before = model.ArenaBatchedDecodeSteps;
        for (int round = 0; round < 7; ++round)
        {
            // Remove and replace a live slot with another request and a distinct
            // prefix. Native registry/flush and pointer recycling must isolate it.
            if (round == 4)
            {
                model.OnSequenceReleased(ids[^1]); reference.OnSequenceReleased(ids[^1]);
                ids[^1] = "newcomer";
                int[] prompt = Prompt(19, 35);
                Assert.True(model.BindSequenceCache(ids[^1])); Assert.True(reference.BindSequenceCache(ids[^1]));
                Assert.Equal(reference.Forward(prompt), model.Forward(prompt));
                Assert.Equal(reference.Forward([197]), model.Forward([197]));
                positions[^1] = prompt.Length + 1;
                model.RestorePrimaryCache(); reference.RestorePrimaryCache();
            }
            var tokens = Enumerable.Range(0, width).Select(i => (37 + 23 * i + 17 * round) % 250).ToArray();
            var expected = new float[width][];
            for (int i = 0; i < width; ++i)
            {
                reference.BindSequenceCache(ids[i]);
                expected[i] = (float[])reference.Forward([tokens[i]]).Clone();
            }
            reference.RestorePrimaryCache();
            if (round == 3)
            {
                // Every holder leaves the arena for one solo step, then rejoins.
                for (int i = 0; i < width; ++i)
                {
                    model.BindSequenceCache(ids[i]);
                    worst = Math.Max(worst, AssertClose(expected[i], model.Forward([tokens[i]]), $"solo {i}"));
                }
                model.RestorePrimaryCache();
            }
            else
            {
                // Vary caller order; output mapping must follow request ids.
                int[] order = Enumerable.Range(0, width).OrderBy(i => (i + round) % width).ToArray();
                var rows = new float[width][];
                Assert.True(model.TryForwardBatchedFusedDecode(order.Select(i => ids[i]).ToArray(),
                    order.Select(i => tokens[i]).ToArray(), order.Select(i => positions[i]).ToArray(), rows),
                    model.BatchedFusedDecodeDeclineReason);
                for (int i = 0; i < width; ++i)
                {
                    worst = Math.Max(worst, AssertClose(expected[order[i]], rows[i], $"round {round}, row {i}"));
                    if (i > 0) Assert.NotSame(rows[0], rows[i]);
                }
            }
            for (int i = 0; i < width; ++i) ++positions[i];
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
        }
        Assert.Equal(6, model.ArenaBatchedDecodeSteps - before);
        // Arena teardown must publish all QSA raw keys, coordinates, recurrent
        // state and KV back to the original holder for a final solo continuation.
        for (int i = 0; i < width; ++i)
        {
            model.BindSequenceCache(ids[i]); reference.BindSequenceCache(ids[i]);
            Assert.Equal(positions[i], model.CacheSeqLen);
            if (qsa)
                Assert.Equal(Field<int[]>(reference, "_qsaPositions").Take(3 * positions[i]),
                    Field<int[]>(model, "_qsaPositions").Take(3 * positions[i]));
            worst = Math.Max(worst, AssertClose(reference.Forward([113]), model.Forward([113]), "final solo " + i));
        }
        output.WriteLine($"{backend}, width={width}, q2kxl={q2kxl}, hostLayers={offloadLayers}: 6 arena steps, growth/media QSA/reorder/churn/GC/solo rejoin; worst |dlogit|={worst:G6}.");
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void PleTranspose_IsOwnedAcrossNewHoldersAndForcedCollections()
        => RunPleLifetime(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void PleTranspose_IsOwnedAcrossNewHoldersAndForcedCollections_Metal()
        => RunPleLifetime(BackendType.GgmlMetal);

    private void RunPleLifetime(BackendType backend)
    {
        string path = Fixture();
        using var model = new Qwen4ExpModel(path, backend);
        model.BindSequenceCache("a"); model.Forward(Prompt(0, 24));
        var transpose = new WeakReference<float[]>(Field<float[]>(model, "_pleConvWT"));
        IntPtr pointer = Field<Qwen4ExpPleArgs[]>(model, "_pleArgs")[0].Conv1dT;
        for (int i = 0; i < 4; ++i)
        {
            model.BindSequenceCache("b-" + i); model.Forward(Prompt(i + 1, 24));
            Assert.Equal(pointer, Field<Qwen4ExpPleArgs[]>(model, "_pleArgs")[0].Conv1dT);
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
            Assert.True(transpose.TryGetTarget(out _), "A live PLE descriptor lost its convolution weight owner.");
        }
        model.BindSequenceCache("a");
        float[] actual = (float[])model.Forward([127]).Clone();
        model.RestorePrimaryCache(); model.ResetKVCache(); model.Forward(Prompt(0, 24));
        Assert.Equal(model.Forward([127]), actual);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public Task ContinuousBatching_ParallelGreedyStreamsMatchSerial_Cpu()
        => RunEngine(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public Task ContinuousBatching_ParallelGreedyStreamsMatchSerial_Metal()
        => RunEngine(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public Task ContinuousBatching_ParallelGreedyStreamsMatchSerial_Cuda()
        => RunEngine(BackendType.GgmlCuda);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(2)] [InlineData(8)]
    public Task HostOffloadedContinuousBatching_MatchesSerial_Metal(int offloadLayers)
        => RunEngine(BackendType.GgmlMetal, offloadLayers);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(2)] [InlineData(8)]
    public Task HostOffloadedContinuousBatching_MatchesSerial_Cuda(int offloadLayers)
        => RunEngine(BackendType.GgmlCuda, offloadLayers);

    private async Task RunEngine(BackendType backend, int offloadLayers = 0)
    {
        string path = Fixture();
        if (offloadLayers > 0) MoeCpuOffloadConfig.SetLayers(offloadLayers);
        using var model = new Qwen4ExpModel(path, backend);
        const int count = 8;
        int[][] prompts = [Prompt(3, 28), Prompt(7, 33), Prompt(11, 39)];
        var expected = new List<int>[prompts.Length];
        var serialLogits = new float[prompts.Length][][];
        foreach (int i in Enumerable.Range(0, prompts.Length))
        {
            model.ResetKVCache();
            float[] logits = null!;
            for (int start = 0; start < prompts[i].Length; start += 8)
                logits = model.Forward(prompts[i].Skip(start).Take(8).ToArray());
            expected[i] = new List<int>();
            serialLogits[i] = new float[count][];
            for (int t = 0; t < count; ++t)
            {
                serialLogits[i][t] = (float[])logits.Clone();
                int token = ArgMax(logits); expected[i].Add(token);
                if (t + 1 < count) logits = model.Forward([token]);
            }
        }
        model.ResetKVCache();
        var config = new SchedulerConfig
        {
            // CUDA greedy parity compares the same eight-token prefill shapes.
            // The aggregate budget also bounds concurrent prefills: a 64-token
            // budget permits larger chunks despite MaxPrefillChunkSize below.
            MaxNumBatchedTokens = backend == BackendType.GgmlCuda ? 8 : 64,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8, SoloPrefillChunkSize = 8,
            NumBlocks = 128, BlockSize = 8, EnablePrefixCaching = false, DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, config, NullLogger.Instance);
        var handles = prompts.Select((prompt, i) => engine.SubmitRequest(new SequenceState(
            "engine-" + i, prompt.ToList(), count, config.BlockSize, SamplingConfig.Greedy))).ToArray();
        var streams = await Task.WhenAll(handles.Select(Drain));
        for (int i = 0; i < streams.Length; ++i)
        {
            for (int t = 0; t < Math.Min(expected[i].Count, streams[i].Count); ++t)
            {
                if (expected[i][t] == streams[i][t]) continue;
                float[] logits = serialLogits[i][t];
                int[] top = Enumerable.Range(0, logits.Length).OrderByDescending(j => logits[j]).Take(5).ToArray();
                output.WriteLine($"Stream {i} first mismatch at {t}: expected={expected[i][t]}, actual={streams[i][t]}, " +
                    $"serial margin={logits[top[0]] - logits[top[1]]:R}, actual token serial score={logits[streams[i][t]]:R}; " +
                    "serial top5=" + string.Join(", ", top.Select(j => $"{j}:{logits[j]:R}")));
                break;
            }
            Assert.Equal(expected[i], streams[i]);
        }
        Assert.True(model.ArenaBatchedDecodeSteps > 0, "Parallel requests never executed a batched graph.");
        output.WriteLine($"{backend}, hostLayers={offloadLayers}: three parallel 8-token streams matched independent serial greedy; {model.ArenaBatchedDecodeSteps} arena steps.");
    }

    private static async Task<List<int>> Drain(InferenceRequestHandle handle)
    {
        var result = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync()) result.Add(token);
        await handle.Completion;
        return result;
    }

    private static int[] Prompt(int seed, int count)
        => Enumerable.Range(0, count).Select(i => (11 + 31 * i + 13 * seed) % 250).ToArray();

    private static int[] MediaAxes(int count)
        => Enumerable.Range(0, count).SelectMany(i => new[] { i / 4, i % 2, i / 2 % 2 }).ToArray();

    private static int ArgMax(float[] values) => Array.IndexOf(values, values.Max());

    private static double AssertClose(float[] expected, float[] actual, string phase)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.All(actual, x => Assert.True(float.IsFinite(x)));
        double difference = expected.Zip(actual, (a, b) => Math.Abs((double)a - b)).Max();
        Assert.True(difference < .03, $"{phase}: max |dlogit|={difference:G8} exceeds 0.03.");
        return difference;
    }

    private static T Field<T>(Qwen4ExpModel model, string name)
        => (T)typeof(Qwen4ExpModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    public void Dispose()
    {
        _environment.Dispose();
        MoeCpuOffloadConfig.Reset();
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
    }
}
