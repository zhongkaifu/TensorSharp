// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

// CPU-only range and shared-storage scheduling gates; never initializes CUDA.
public sealed class Qwen35VerifyDecodeQ81Tests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(16)]
    [InlineData(64)]
    public void PureVerifyOrDecode_CoversExactlyEveryRowWithNativeSupportedChunks(int rows)
    {
        var plan = Q35CudaEngine.PlanVerifyDecodeQ81(new[] { rows }, new[] { true });
        Assert.Equal(rows, plan.TotalRows);
        Assert.Equal(rows, plan.ForcedRows);
        Assert.Equal(rows > 0, plan.AllForced);
        Assert.Equal(rows > 8, plan.NeedsOverride);
        var actual = plan.ForcedRanges.SelectMany(Q35CudaEngine.Q81Chunks).ToArray();
        Assert.All(actual, range => Assert.InRange(range.Rows, 1, 8));
        Assert.Equal(Enumerable.Range(0, rows), actual.SelectMany(r => Enumerable.Range(r.Start, r.Rows)));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(4095)]
    [InlineData(4096)]
    public void Mixed_PreservesOriginalFullNGeometryAndOverridesOnlyVerifyTail(int prefill)
    {
        var plan = Q35CudaEngine.PlanVerifyDecodeQ81(new[] { prefill, 64 }, new[] { false, true });
        Assert.False(plan.AllForced);
        Assert.True(plan.NeedsOverride);
        Assert.Equal(prefill + 64, plan.TotalRows); // Original A4 receives this, never just prefill.
        var chunks = plan.ForcedRanges.SelectMany(Q35CudaEngine.Q81Chunks).ToArray();
        Assert.Equal(Enumerable.Range(prefill, 64), chunks.SelectMany(r => Enumerable.Range(r.Start, r.Rows)));
        Assert.DoesNotContain(chunks, r => r.Start < prefill);
    }

    [Fact]
    public void MultipleGroupsAndRaggedWindows_HaveNoGapsOverlapOrPrefillWrites()
    {
        var plan = Q35CudaEngine.PlanVerifyDecodeQ81(new[] { 31, 7, 3, 14 }, new[] { false, true, false, true });
        Assert.Equal(55, plan.TotalRows);
        Assert.Equal(21, plan.ForcedRows);
        var selected = plan.ForcedRanges.SelectMany(Q35CudaEngine.Q81Chunks)
            .SelectMany(r => Enumerable.Range(r.Start, r.Rows)).ToArray();
        Assert.Equal(Enumerable.Range(31, 7).Concat(Enumerable.Range(41, 14)), selected);
        Assert.Equal(selected.Length, selected.Distinct().Count());
    }

    [Fact]
    public void PlainPrefill_IncludingOneRowTails_IsNeverForcedByItsSize()
    {
        foreach (int rows in new[] { 0, 1, 8, 9, 4096 })
        {
            var plan = Q35CudaEngine.PlanVerifyDecodeQ81(new[] { rows }, new[] { false });
            Assert.False(plan.NeedsOverride);
            Assert.False(plan.AllForced);
            Assert.Empty(plan.ForcedRanges);
        }
    }

    [Fact]
    public void EveryProjectionUsesItsOwnInputAndOutputStride_AtMixed4160Tail()
    {
        var plan = Q35CudaEngine.PlanVerifyDecodeQ81(new[] { 4096, 64 }, new[] { false, true });
        // Actual 27B dimensions: attention/GDN InProj, GDN/attention OutProj,
        // interleaved gate|up and FFN down. Each output uses w.Rows, not hidden width.
        foreach (var (k, outDim) in new[] { (5120, 14336), (5120, 16480), (6144, 5120), (5120, 34816), (17408, 5120) })
            foreach (var chunk in plan.ForcedRanges.SelectMany(Q35CudaEngine.Q81Chunks))
            {
                long input = Q35CudaEngine.ProjectionRowByteOffset(chunk.Start, k);
                long output = Q35CudaEngine.ProjectionRowByteOffset(chunk.Start, outDim);
                Assert.True(input >= 4096L * k * 4 && input + chunk.Rows * (long)k * 4 <= 4160L * k * 4);
                Assert.True(output >= 4096L * outDim * 4 && output + chunk.Rows * (long)outDim * 4 <= 4160L * outDim * 4);
            }
    }

    [Fact]
    public void GateUp_ConsumesAllNormalizedViewsBeforeAnyActivationWrite()
    {
        var mixed = Q35CudaEngine.Q81GateStages(false);
        var all = Q35CudaEngine.Q81GateStages(true);
        Assert.Equal(new[] { Q35CudaEngine.Q81GateStage.ProjectQ81ToProj, Q35CudaEngine.Q81GateStage.OriginalA4FusedToAct, Q35CudaEngine.Q81GateStage.SwiGluQ81ToAct }, mixed);
        Assert.Equal(new[] { Q35CudaEngine.Q81GateStage.ProjectQ81ToProj, Q35CudaEngine.Q81GateStage.SwiGluQ81ToAct }, all);
        // Stress valid slab geometry where xn and act DO overlap, including across
        // different row indices. The real 27B widths are checked separately below.
        var slab = Q35CudaEngine.PlanSharedScratch(65, 32, 1, 3, 1, 2, 17);
        var xn = Assert.Single(slab.Spans, s => s.Name == "_xn");
        var act = Assert.Single(slab.Spans, s => s.Name == "_act");
        bool Overlap(long a, long an, long b, long bn) => a < b + bn && b < a + an;
        Assert.True(Overlap(xn.Offset, xn.Bytes, act.Offset, act.Bytes));
        bool LiveOrder(IReadOnlyList<Q35CudaEngine.Q81GateStage> stages)
        {
            bool xnUnread = true;
            foreach (var stage in stages)
            {
                if (stage == Q35CudaEngine.Q81GateStage.ProjectQ81ToProj) xnUnread = false;
                else if (xnUnread && Overlap(xn.Offset, xn.Bytes, act.Offset, act.Bytes)) return false;
            }
            return !xnUnread;
        }
        Assert.True(LiveOrder(mixed));
        Assert.True(LiveOrder(all));
        Assert.False(LiveOrder(new[] { mixed[1], mixed[0], mixed[2] })); // Catches the tempting unsafe order.
        var actual = Q35CudaEngine.PlanSharedScratch(4160, 5120, 6144, 10240, 48, 6144, 17408);
        var actualXn = Assert.Single(actual.Spans, s => s.Name == "_xn");
        var actualAct = Assert.Single(actual.Spans, s => s.Name == "_act");
        Assert.False(Overlap(actualXn.Offset, actualXn.Bytes, actualAct.Offset, actualAct.Bytes));
    }

    [Fact]
    public void InvalidAndOverflowingRanges_FailBeforeAnyPointerCalculationOrDispatch()
    {
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.PlanVerifyDecodeQ81(new[] { 1 }, Array.Empty<bool>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.PlanVerifyDecodeQ81(new[] { -1 }, new[] { true }));
        Assert.Throws<OverflowException>(() => Q35CudaEngine.PlanVerifyDecodeQ81(new[] { int.MaxValue, 1 }, new[] { false, true }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.Q81Chunks(new(-1, 8)));
        Assert.Throws<OverflowException>(() => Q35CudaEngine.Q81Chunks(new(int.MaxValue, 8)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.ProjectionRowByteOffset(-1, 5120));
        Assert.Throws<OverflowException>(() => Q35CudaEngine.ProjectionRowByteOffset(int.MaxValue, int.MaxValue));
    }


    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ActualEnginePolicy_IsolatesModelAndPrefixNamespacesWithoutChangingLegacyBytes(bool q81, bool trace)
    {
        var engine = MetadataOnlyEngine(q81, trace);
        Assert.False(engine.IsNinferModel);
        var executor = (Qwen35CudaExecutor)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35CudaExecutor));
        SetField(executor, "_engine", engine);
        var model = (Qwen35CudaModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35CudaModel));
        SetField(model, "_exec", executor);
        var config = new ModelConfig { NumLayers = 64, HiddenSize = 5120, VocabSize = 248320 };
        SetBaseField(model, "<Config>k__BackingField", config);
        SetBaseField(model, "_maxContextLength", 262144);
        string suffix = q81 ? "|verify-decode=q8_1-v1" : string.Empty;
        const string legacy = "qwen35|exec=cuda|L=64|H=5120|V=248320|ctx=262144";

        Assert.Equal(q81, engine.VerifyDecodeQ81Enabled);
        Assert.Equal(suffix, engine.KVStateNamespaceSuffix);
        Assert.Equal(legacy + suffix, model.KVStateFingerprint);
        Assert.Equal(model.KVStateFingerprint, model.GetPrefixCacheCapabilities().NamespaceFingerprint);
        if (!q81) Assert.Equal(System.Text.Encoding.UTF8.GetBytes(legacy), System.Text.Encoding.UTF8.GetBytes(model.KVStateFingerprint));

        // Keep the model's existing live metadata contract: adding a policy suffix
        // must not freeze the rest of the fingerprint at engine construction time.
        config.NumLayers = 2;
        config.HiddenSize = 128;
        config.VocabSize = 256;
        SetBaseField(model, "_maxContextLength", 4096);
        Assert.Equal("qwen35|exec=cuda|L=2|H=128|V=256|ctx=4096" + suffix, model.KVStateFingerprint);
        Assert.Equal(model.KVStateFingerprint, model.GetPrefixCacheCapabilities().NamespaceFingerprint);
    }

    [Fact]
    public void UnloadedModel_PreservesItsExistingNamespaceWithoutAnExecutor()
    {
        var model = (Qwen35CudaModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35CudaModel));
        SetBaseField(model, "<Config>k__BackingField", new ModelConfig { NumLayers = 2, HiddenSize = 128, VocabSize = 256 });
        SetBaseField(model, "_maxContextLength", 4096);
        Assert.Equal("qwen35|exec=cuda|L=2|H=128|V=256|ctx=4096", model.KVStateFingerprint);
        Assert.Equal(model.KVStateFingerprint, model.GetPrefixCacheCapabilities().NamespaceFingerprint);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TraceOff_RecordsNothingAndAllocatesNoInstrumentationEvenWithQ81Enabled(bool q81)
    {
        var engine = MetadataOnlyEngine(q81, trace: false);
        for (int i = 0; i < 1000; i++) ExerciseDisabledInstrumentation(engine);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10000; i++) ExerciseDisabledInstrumentation(engine);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated); // CPU instrumentation only, not a GPU/full-forward allocation claim.
        Assert.Null(typeof(Q35CudaEngine).GetField("_projectionDispatches", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(engine));
        Assert.Same(Array.Empty<Q35CudaEngine.ProjectionDispatchStat>(), engine.ProjectionDispatchStats);
        Assert.Equal(0, engine.ForcedQ81ProjectionCalls);
        Assert.Equal(0, engine.ForcedQ81ProjectionRows);
    }

    [Fact]
    public void TraceOn_LazilyCountsActualProjectionRangesAndResetPreservesCumulativeForcedTotals()
    {
        var engine = MetadataOnlyEngine(q81: true, trace: true);
        Assert.Null(typeof(Q35CudaEngine).GetField("_projectionDispatches", BindingFlags.NonPublic | BindingFlags.Instance)!.GetValue(engine));
        Assert.Empty(engine.ProjectionDispatchStats);
        engine.RecordProjectionDispatch("InProj", "NVFP4_W4A4", 0, 4160);
        engine.RecordProjectionDispatch("InProj", "NVFP4_W4A4", 0, 4160);
        engine.RecordForcedQ81Projection("InProj", 4096, 8);
        engine.RecordForcedQ81Projection("InProj", 4096, 8);
        engine.RecordForcedQ81Projection("GateUp", 4104, 3);
        var stats = engine.ProjectionDispatchStats;
        Assert.Equal(3, stats.Length);
        Assert.Contains(new Q35CudaEngine.ProjectionDispatchStat("InProj", "NVFP4_W4A4", 0, 4160, 2), stats);
        Assert.Contains(new Q35CudaEngine.ProjectionDispatchStat("InProj", "NVFP4_W4A8_Q8_1", 4096, 8, 2), stats);
        Assert.Contains(new Q35CudaEngine.ProjectionDispatchStat("GateUp", "NVFP4_W4A8_Q8_1", 4104, 3, 1), stats);
        Assert.Equal(3, engine.ForcedQ81ProjectionCalls);
        Assert.Equal(19, engine.ForcedQ81ProjectionRows);

        engine.ResetProjectionDispatchStats();
        Assert.Same(Array.Empty<Q35CudaEngine.ProjectionDispatchStat>(), engine.ProjectionDispatchStats);
        engine.RecordForcedQ81Projection("Down", 4104, 3);
        Assert.Equal(new Q35CudaEngine.ProjectionDispatchStat("Down", "NVFP4_W4A8_Q8_1", 4104, 3, 1), Assert.Single(engine.ProjectionDispatchStats));
        Assert.Equal(4, engine.ForcedQ81ProjectionCalls);
        Assert.Equal(22, engine.ForcedQ81ProjectionRows);
    }

    private static void ExerciseDisabledInstrumentation(Q35CudaEngine engine)
    {
        engine.RecordProjectionDispatch("InProj", "NVFP4_W4A4", 0, 4160);
        engine.RecordForcedQ81Projection("GateUp", 4096, 8);
        engine.ResetProjectionDispatchStats();
        _ = engine.ProjectionDispatchStats;
    }

    // Exercise actual engine/model property and instrumentation implementations
    // without running constructors, loading weights or creating a CUDA context.
    private static Q35CudaEngine MetadataOnlyEngine(bool q81, bool trace)
    {
        var engine = (Q35CudaEngine)RuntimeHelpers.GetUninitializedObject(typeof(Q35CudaEngine));
        // The production constructor always supplies a nonnull model recipe.
        SetField(engine, "_m", new Q35CudaEngine.ModelDesc { NLayer = 64, NEmbd = 5120, NVocab = 248320, NCtx = 262144 });
        SetField(engine, "_verifyDecodeQ81", q81);
        SetField(engine, "_projectionTrace", trace);
        return engine;
    }

    private static void SetField(object instance, string name, object value) => instance.GetType()
        .GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.SetValue(instance, value);
    private static void SetBaseField(ModelBase model, string name, object value) => typeof(ModelBase)
        .GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)!.SetValue(model, value);

    [Fact]
    public void ProjectionPolicyGraphGate_RejectsEitherGraphAndKeepsOffPolicyUnchanged()
    {
        Q35CudaEngine.ValidateVerifyDecodeQ81Graphs(false, true, true);
        Q35CudaEngine.ValidateVerifyDecodeQ81Graphs(true, false, false);
        Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateVerifyDecodeQ81Graphs(true, true, false));
        Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateVerifyDecodeQ81Graphs(true, false, true));
    }
}
