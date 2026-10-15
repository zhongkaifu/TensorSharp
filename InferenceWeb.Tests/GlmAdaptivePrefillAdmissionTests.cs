// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Paged;

namespace InferenceWeb.Tests;

/// <summary>Metadata, actual quota transactions and fake fenced-native lifecycle
/// checks. No GPU allocator or model performance acceptance is implied.</summary>
public sealed class GlmAdaptivePrefillAdmissionTests
{
    private const long MiB = 1L << 20, GiB = 1L << 30;
    private const string Host = UnifiedMemoryModelSession.HostPool, Device = UnifiedMemoryModelSession.DevicePool;
    private static MemoryCharge[] Charges(long host, long device) => [new(Host, host), new(Device, device)];
    private static MemoryCharge[] DeviceCharges(long bytes) => [new(Device, bytes)];
    private static InferenceMemoryBytes Bytes(BudgetReservation owner)
        => new(owner.OwnedCharges.Single(c => c.Pool == Host).Bytes, owner.OwnedCharges.Single(c => c.Pool == Device).Bytes);

    private static GlmUnifiedMemoryGeometry Geometry()
    {
        GlmUnifiedExpertLayout Layout(long bytes, GgmlTensorType down) => new(
            new(GgmlTensorType.IQ1_S, bytes / 3, 0, true), new(GgmlTensorType.IQ1_S, bytes / 3, 0, true),
            new(down, bytes - 2 * (bytes / 3), 0, true));
        GlmUnifiedExpertLayout[] groups = [Layout(7_536_640, GgmlTensorType.IQ3_XXS),
            Layout(6_488_064, GgmlTensorType.IQ4_XS), Layout(8_781_824, GgmlTensorType.IQ2_XXS)];
        return new() { Layers = 45, DeclaredContextTokens = 32768, RecurrentLayers = 34,
            Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
            KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8,
            ExpertCount = 288, DenseDeviceBytes = 6_592_982_264,
            ExpertLayerBytes = Enumerable.Repeat(30 * GiB, 45).ToArray(),
            ExpertLayerLayouts = Enumerable.Range(0, 45).Select(i => groups[i % 3]).ToArray() };
    }

    [Theory]
    [InlineData(1)]
    [InlineData(128)]
    [InlineData(512)]
    [InlineData(1024)]
    public void SerialOutputIsOneLogitsRowWhileJointDecodePreservesEveryOutputRow(int rows)
    {
        var geometry = Geometry();
        var serial = geometry.Workspace(1536, rows, 1);
        var allRows = geometry.Workspace(1536, rows, rows);
        long delta = 4L * geometry.Vocab * (rows - 1);
        Assert.Equal(delta, allRows.Host - serial.Host);
        Assert.Equal(delta, allRows.Device - serial.Device);
        Assert.Equal(2 * serial.Host, geometry.GraphPeak(1536, rows, false).Host);
        Assert.Equal(2 * serial.Device, geometry.GraphPeak(1536, rows, false).Device);
        Assert.Equal(2 * allRows.Host, geometry.GraphPeak(1536, rows, true).Host);
        Assert.Equal(2 * (allRows.Device + (rows - 1L) * geometry.FixedStateBytes),
            geometry.GraphPeak(1536, rows, true).Device);
        Assert.Equal(709_230_592L, geometry.SlotBytes(32768));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(513)]
    public void InvalidOutputRowsCannotReduceTheDeclaredWorkspace(int outputs)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Geometry().Workspace(1536, 512, outputs));

    [Fact]
    public void AdaptiveLoadProtectsEveryPhysicalSlotAndFullContextJointDecodeButAdmitsWidePrefillSeparately()
    {
        var geometry = Geometry();
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true };
        var adaptive = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, 14 * GiB, 0);
        var ordinary = GlmSelectedExpertMemoryPlan.SelectForSession(geometry,
            options with { EnableAdaptiveGlmPrefill = false }, 32768, 14 * GiB, 0);
        Assert.Equal(512, adaptive.PrefillRows);
        Assert.InRange(ordinary.PrefillRows, 1, 511);
        Assert.Equal(45, adaptive.CpuExpertLayers);
        Assert.Equal(geometry.AdaptiveExecutionPlacementReserve(32768, 3) + adaptive.CacheCapacityBytes,
            adaptive.ExecutionReserveBytes);
        Assert.True(adaptive.ExecutionReserveBytes + geometry.DenseDeviceBytes <= 14 * GiB);
        Assert.True(adaptive.CacheMinimumForecastBytes <= adaptive.CacheCapacityBytes);
        Assert.Equal(GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 512), adaptive.CacheMinimumForecastBytes);
        Assert.True(geometry.SlotBytes(32768) * 3 < adaptive.ExecutionReserveBytes - adaptive.CacheCapacityBytes);
        Assert.False(new UnifiedMemoryModelOptions().EnableAdaptiveGlmPrefill);
    }

    [Fact]
    public void AdaptiveLoadCannotOmitVisionTheExpertFloorOrTheConcurrentSlotPlacement()
    {
        var geometry = Geometry();
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true };
        long floor = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1);
        long baseReserve = geometry.AdaptiveExecutionPlacementReserve(32768, 3);
        long exact = geometry.DenseDeviceBytes + baseReserve + floor;
        Assert.Equal(3, GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, exact, 0).PrefillRows);
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, exact - 1, 0));
        Assert.Throws<MemoryPressureException>(() => GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, exact, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.SelectForSession(geometry,
            options with { EnableGlmSelectedExperts = false }, 32768, 14 * GiB, 0));
    }

    [Fact]
    public void NullableAdaptiveGlmBoundPreservesApiCommonDefaultsAndOnlyItsEnabledPolicyUsesTheOverride()
    {
        var ordinary = new UnifiedMemoryModelOptions();
        Assert.Null(ordinary.AdaptiveGlmPrefillChunkTokens);
        Assert.Equal(512, ordinary.ResolveGlmPrefillChunkTokens());
        Assert.Equal(128, (ordinary with { PrefillChunkTokens = 128, EnableAdaptiveGlmPrefill = true }).ResolveGlmPrefillChunkTokens());
        Assert.Equal(512, (ordinary with { AdaptiveGlmPrefillChunkTokens = 1024 }).ResolveGlmPrefillChunkTokens());
        Assert.Equal(1024, (ordinary with { EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024 }).ResolveGlmPrefillChunkTokens());
        Assert.Throws<ArgumentOutOfRangeException>(() => (ordinary with { PrefillChunkTokens = 1024,
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024 }).ResolveGlmPrefillChunkTokens());
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(0)]
    [InlineData(1025)]
    public void InvalidAdaptiveGlmBoundsRefuseBeforeAnyModelOrNativePlanning(int rows)
    {
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true,
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = rows };
        Assert.Throws<ArgumentOutOfRangeException>(() => options.ResolveGlmPrefillChunkTokens());
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.SelectForSession(
            Geometry(), options, 32768, 20 * GiB, 0));
    }

    [Theory]
    [InlineData(8, 1024)]
    [InlineData(16, 512)]
    [InlineData(64, 128)]
    public void AdaptiveGlmRouteBoundIsIndependentOfUnchangedLegacyPolicy(int top, int cap)
    {
        var geometry = Geometry() with { UsedExperts = top };
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true,
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024 };
        var wide = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, 30 * GiB, 0);
        Assert.Equal(cap, wide.PrefillRows);
        Assert.InRange(wide.PrefillRows * top, 1, 8192);
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, cap + 1));
        var ordinary = GlmSelectedExpertMemoryPlan.SelectForSession(geometry,
            options with { EnableAdaptiveGlmPrefill = false }, 32768, 30 * GiB, 0);
        Assert.Equal(Math.Min(512, 4096 / top), ordinary.PrefillRows);
        Assert.Equal(512, options.PrefillChunkTokens);
    }

    [Fact]
    public void WiderImmutablePhaseFloorCanReduceTheLoadCeilingWithoutBorrowingPhysicalSlotPlacement()
    {
        var geometry = Geometry();
        long fullWidthFloor = GlmSelectedExpertMemoryPlan.ForecastMinimum(geometry, 3, 1024);
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true, EnableAdaptiveGlmPrefill = true,
            AdaptiveGlmPrefillChunkTokens = 1024, MaximumExpertCacheBytes = fullWidthFloor - 1 };
        var plan = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, 32768, 20 * GiB, 0);
        Assert.Equal(1023, plan.PrefillRows);
        Assert.Equal(fullWidthFloor - 1, plan.CacheCapacityBytes);
        Assert.True(plan.CacheMinimumForecastBytes <= plan.CacheCapacityBytes);
        Assert.Equal(geometry.AdaptiveExecutionPlacementReserve(32768, 3) + plan.CacheCapacityBytes, plan.ExecutionReserveBytes);
        Assert.Equal(709_230_592, geometry.SlotBytes(32768));
        Assert.Equal(1024, options.AdaptiveGlmPrefillChunkTokens);
    }

    [Fact]
    public void ActualPromptWideAdmissionKeepsBothGraphGenerationsFullRequestSlotAndPhaseFloor()
    {
        var geometry = Geometry();
        var budget = new MemoryBudget(Charges(8 * GiB, 16 * GiB));
        using var dense = budget.Reserve(Charges(0, 7 * GiB)); dense.Commit();
        using var experts = budget.Reserve(DeviceCharges(4 * GiB));
        var oldPeak = geometry.GraphPeak(256, 17, false);
        using var joint = budget.Reserve(Charges(oldPeak.Host, oldPeak.Device));
        var old = geometry.GraphPeak(256, 17, false, 1);
        using var child = joint.TryTake(Charges(old.Host, old.Device))!; child.Commit();
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(8 * GiB, 16 * GiB),
            Bytes(joint), default, 4 * GiB, 3 * GiB, 1024, rows =>
            {
                var peak = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, rows, 1340, 0, [], 2).MaximumPeak;
                return (GlmUnifiedPrefillAdmission.PreserveReplacement(peak, old, 2),
                    new(geometry.Workspace(1340, rows, 1).Host, geometry.SlotBytes(32768)));
            }, out var selected));
        Assert.InRange(selected.Rows, 513, 1023);
        Assert.True(selected.ExpertTargetBytes >= 3 * GiB);
        Assert.Equal(geometry.GraphPeak(1536, selected.Rows, false, 2), selected.JointTarget);
        Assert.Equal(geometry.SlotBytes(32768), selected.RequestTarget.Device);
        Assert.Equal(old.Device, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(4 * GiB, experts.OwnedCharges.Single().Bytes); // Selection does not fake physical reclamation.
    }

    [Fact]
    public void WiderGeometryChargesTheFullGlmRouteScratchAndNeverClaimsFullContextSinglePassFits()
    {
        var geometry = Geometry();
        Assert.Equal(32768, GlmUnifiedMemoryGeometry.SelectedRouteScratchHostBytes);
        Assert.Equal(196608, GlmUnifiedMemoryGeometry.SelectedRouteMetadataHostBytes);
        var shape = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 1024, 1340, 0, [32000], 2);
        Assert.Equal(1024, shape.PrefillRows);
        Assert.Equal(1536, shape.SerialContext);
        Assert.Equal(32256, shape.JointContext);
        Assert.Equal(1_423_566_848, shape.PrefillPeak.Host);
        Assert.Equal(5_947_844_608, shape.PrefillPeak.Device);
        Assert.True(geometry.GraphPeak(32768, 1024, false).Device > shape.PrefillPeak.Device);
        Assert.Equal(709_230_592L, geometry.SlotBytes(32768));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedPrefillAdmission.TrySelect(
            [new(Host, GiB, 0, 0), new(Device, GiB, 0, 0)], Charges(GiB, GiB), default, default,
            0, 0, 1025, _ => (default, default), out _));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(6)]
    [InlineData(8)]
    public void OldOrUnknownSelectedExpertAbiCannotSilentlyUseTheWiderManagedPlan(int version)
        => Assert.Throws<NotSupportedException>(() => GgmlGlmNative.ValidateSelectedExpertAdapterVersion(version));

    [Fact]
    public void MatchingSelectedExpertAbiPermitsTheWiderPolicyWithoutLoadingANativeModel()
        => GgmlGlmNative.ValidateSelectedExpertAdapterVersion(7);

    [Fact]
    public void SelectionPrefersWidePrefillByReclaimingOnlyExpertSurplusWithoutAnyMutation()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var dense = budget.Reserve(Charges(0, 100 * MiB));
        using var expert = budget.Reserve(DeviceCharges(400 * MiB));
        using var joint = budget.Reserve(Charges(100 * MiB, 100 * MiB));
        var before = budget.Snapshot();
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(before, Charges(1000 * MiB, 1000 * MiB),
            Bytes(joint), default, 400 * MiB, 200 * MiB, 512,
            rows => (new(100 * MiB, rows * MiB), new(100 * MiB, 100 * MiB)), out var chosen));
        Assert.Equal(512, chosen.Rows);
        Assert.Equal(200 * MiB, chosen.ExpertTargetBytes);
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void RamPressureChoosesSmallerRowsAndCannotBorrowCudaCreditForHostMemory()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 5000 * MiB));
        using var expert = budget.Reserve(DeviceCharges(400 * MiB));
        using var joint = budget.Reserve(Charges(100 * MiB, 100 * MiB));
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(200 * MiB, 5000 * MiB),
            Bytes(joint), default, 400 * MiB, 200 * MiB, 512,
            rows => (new(rows * MiB, 100 * MiB), new(100 * MiB, 100 * MiB)), out var chosen));
        Assert.Equal(100, chosen.Rows);
        Assert.Equal(400 * MiB, chosen.ExpertTargetBytes);
    }

    [Fact]
    public void FreshPhysicalPressureAndImmutableFloorCanRefuseEveryWidth()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var dense = budget.Reserve(Charges(0, 100 * MiB));
        using var expert = budget.Reserve(DeviceCharges(400 * MiB));
        using var joint = budget.Reserve(Charges(100 * MiB, 100 * MiB));
        var before = budget.Snapshot();
        Assert.False(GlmUnifiedPrefillAdmission.TrySelect(before, Charges(1000 * MiB, 500 * MiB),
            Bytes(joint), default, 400 * MiB, 300 * MiB, 512,
            rows => (new(100 * MiB, 100 * MiB + rows * MiB), new(0, 100 * MiB)), out _));
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
    }

    [Fact]
    public void ConfirmedColdPostLookupMayReclaimUnrelatedRetainedStateOnlyForADegradedWidth()
    {
        Assert.True(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(true, 0, 0, 128, 512));
        Assert.False(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(false, 0, 0, 128, 512));
        Assert.False(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(true, 0, 0, 512, 512));
    }

    [Fact]
    public void FollowupLookupHitPreservesItsValidTerminalStateEvenUnderWidthPressure()
    {
        Assert.False(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(true, 1336, 1336, 128, 512));
        Assert.False(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(true, 1336, 0, 128, 512));
        Assert.False(GlmUnifiedPrefillAdmission.ShouldReclaimRetained(true, 0, 1336, 128, 512));
    }

    private static SequenceState Sequence(string id, int prompt, int computed, SequenceStatus status = SequenceStatus.Running)
    {
        var sequence = new SequenceState(id, Enumerable.Repeat(1, prompt).ToArray(), 32768 - prompt, 16, SamplingConfig.Greedy)
            { Status = status };
        // Metadata-only block ownership; no backing KV storage or model call.
        for (int index = 0; index < (computed + 15) / 16; ++index)
            sequence.BlockTable.AppendBlock(new KvBlock(index));
        for (int index = prompt; index < computed; ++index) sequence.AppendOutputToken(1);
        sequence.AdvanceComputedTokens(computed);
        return sequence;
    }

    [Fact]
    public void CancellingLargePrefillReturnsUnusedSharedCreditWithoutLosingTheSurvivorsReusableDecodeGraph()
    {
        var geometry = Geometry();
        var large = geometry.GraphPeak(1536, 512, false);
        var survivor = Sequence("survivor", 17, 17);
        var canceled = Sequence("canceled", 1340, 1024, SequenceStatus.FinishedAborted);
        var decodeChild = geometry.GraphPeak(256, 1, false, 1);
        var budget = new MemoryBudget(Charges(8 * GiB, 16 * GiB));
        using var joint = budget.Reserve(Charges(large.Host, large.Device));
        using var child = joint.TryTake(Charges(decodeChild.Host, decodeChild.Device))!; child.Commit();
        var before = budget.Snapshot().Single(p => p.Pool == Device).Available;
        int canceledPromiseReads = 0;
        var plan = GlmUnifiedReleaseGraphPlan.Create(geometry, 32768, [survivor, canceled],
            seq => { if (ReferenceEquals(seq, canceled)) { canceledPromiseReads++; return large; } return default; },
            decodeChild, 2);
        Assert.Equal(1, plan.RunningRows); Assert.False(plan.HasRunningPrefill);
        Assert.Equal(0, canceledPromiseReads);
        Assert.Equal(geometry.GraphPeak(256, 1, false), plan.Target);
        Assert.True(joint.TryResizeTotal(Charges(plan.Target.Host, plan.Target.Device)));
        Assert.Equal(decodeChild.Device, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        long returned = budget.Snapshot().Single(p => p.Pool == Device).Available - before;
        Assert.Equal(large.Device - plan.Target.Device, returned);
        Assert.True(returned > 512 * MiB);
    }

    [Fact]
    public void ReleasePlanUsesEveryRemainingHeadAndTheActualJointDecodeWidth()
    {
        var geometry = Geometry();
        var plan = GlmUnifiedReleaseGraphPlan.Create(geometry, 32768,
            [Sequence("short", 17, 3000), Sequence("long", 17, 31999)], _ => default, default, 2);
        Assert.Equal(2, plan.RunningRows); Assert.False(plan.HasRunningPrefill);
        Assert.Equal(geometry.GraphPeak(32000, 2, true), plan.Target);
    }

    [Fact]
    public void SurvivingPrefillPromiseAndPhysicalReplacementChildrenCannotBecomeCacheCredit()
    {
        var geometry = Geometry();
        var pending = geometry.GraphPeak(8192, 384, false);
        var actual = new InferenceMemoryBytes(pending.Host + MiB, pending.Device + MiB);
        var plan = GlmUnifiedReleaseGraphPlan.Create(geometry, 32768,
            [Sequence("prefill", 8192, 512), Sequence("decode", 17, 17)], _ => pending, actual, 2);
        Assert.True(plan.HasRunningPrefill); Assert.Equal(2, plan.RunningRows);
        Assert.Equal(actual.Host + pending.Host / 2, plan.Target.Host);
        Assert.Equal(actual.Device + pending.Device / 2, plan.Target.Device);
        var budget = new MemoryBudget(Charges(8 * GiB, 16 * GiB));
        using var joint = budget.Reserve(Charges(plan.Target.Host + MiB, plan.Target.Device + MiB));
        using var child = joint.TryTake(Charges(actual.Host, actual.Device))!; child.Commit();
        Assert.True(joint.TryResizeTotal(Charges(plan.Target.Host, plan.Target.Device)));
        Assert.Equal(actual.Device, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.True(joint.OwnedCharges.Single(c => c.Pool == Device).Bytes >= pending.Device);
    }

    [Fact]
    public void ZeroRunningReleasesOnlyUnusedCreditAndKeepsUnretiredGraphChildrenCharged()
    {
        var geometry = Geometry();
        var budget = new MemoryBudget(Charges(8 * GiB, 16 * GiB));
        using var joint = budget.Reserve(Charges(1000 * MiB, 3000 * MiB));
        using var retainedSlot = budget.Reserve(Charges(0, geometry.SlotBytes(32768)));
        using var child = joint.TryTake(Charges(100 * MiB, 200 * MiB))!; child.Commit();
        var plan = GlmUnifiedReleaseGraphPlan.Create(geometry, 32768, [],
            _ => throw new InvalidOperationException("no live prefill owner"), new(100 * MiB, 200 * MiB), 2);
        Assert.Equal(0, plan.RunningRows); Assert.Equal(new(100 * MiB, 200 * MiB), plan.Target);
        Assert.True(joint.TryResizeTotal(Charges(plan.Target.Host, plan.Target.Device)));
        Assert.Equal(200 * MiB, joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        child.Dispose(); // Simulated successful fenced graph retirement.
        var empty = GlmUnifiedReleaseGraphPlan.Create(geometry, 32768, [], _ => default, default, 2);
        Assert.Equal(default, empty.Target);
        Assert.True(joint.TryResizeTotal(Charges(0, 0)));
        Assert.Equal(0, joint.OwnedCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(geometry.SlotBytes(32768), retainedSlot.OwnedCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Fact]
    public void InvalidLiveReleaseHeadCannotAuthorizeAClampedGraph()
        => Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedReleaseGraphPlan.Create(Geometry(), 32768,
            [Sequence("overflow", 17, 32768)], _ => default, default, 2));

    [Fact]
    public void MaterializedChildrenAndAnotherPrefillPromiseStayOwnedInTheAtomicTransaction()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var joint = budget.Reserve(Charges(200 * MiB, 400 * MiB));
        using var request = budget.Reserve(Charges(200 * MiB, 300 * MiB));
        using var jointChild = joint.TryTake(Charges(100 * MiB, 350 * MiB))!; jointChild.Commit();
        using var slot = request.TryTake(Charges(50 * MiB, 250 * MiB))!; slot.Commit();
        var peerPromise = new InferenceMemoryBytes(200 * MiB, 400 * MiB);
        var childFloor = new InferenceMemoryBytes(50 * MiB, 250 * MiB);
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(1000 * MiB, 1000 * MiB),
            Bytes(joint), Bytes(request), 0, 0, 512,
            rows => (GlmUnifiedPrefillAdmission.Floor(new(100 * MiB, rows * MiB), peerPromise),
                GlmUnifiedPrefillAdmission.Floor(new(50 * MiB, 0), childFloor)), out var chosen));
        Assert.True(chosen.JointTarget.Device >= peerPromise.Device);
        Assert.Equal(250 * MiB, chosen.RequestTarget.Device);
        Assert.True(budget.TryResizeTotals([(joint, Charges(chosen.JointTarget.Host, chosen.JointTarget.Device)),
            (request, Charges(chosen.RequestTarget.Host, chosen.RequestTarget.Device))],
            Charges(1000 * MiB, 1000 * MiB), requirePhysicalFit: true));
        Assert.Equal(350 * MiB, joint.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
        Assert.Equal(250 * MiB, request.LiveChildCharges.Single(c => c.Pool == Device).Bytes);
    }

    [Fact]
    public void CompetingOwnerAfterSelectionMakesTheTransactionRefuseWithoutPartialCreditChanges()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var joint = budget.Reserve(Charges(100 * MiB, 100 * MiB));
        using var request = budget.Reserve(Charges(100 * MiB, 100 * MiB));
        Assert.True(GlmUnifiedPrefillAdmission.TrySelect(budget.Snapshot(), Charges(1000 * MiB, 1000 * MiB),
            Bytes(joint), Bytes(request), 0, 0, 512,
            rows => (new(100 * MiB, 700 * MiB), new(100 * MiB, 200 * MiB)), out var chosen));
        using var competitor = budget.Reserve(DeviceCharges(200 * MiB));
        var before = budget.Snapshot();
        Assert.False(budget.TryResizeTotals([(joint, Charges(chosen.JointTarget.Host, chosen.JointTarget.Device)),
            (request, Charges(chosen.RequestTarget.Host, chosen.RequestTarget.Device))],
            Charges(1000 * MiB, 1000 * MiB), requirePhysicalFit: true));
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
        Assert.Equal(100 * MiB, Bytes(joint).Device);
        Assert.Equal(100 * MiB, Bytes(request).Device);
    }

    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        internal int Updates;
        internal long Budget;
        internal Action<long>? OnUpdate;
        public ulong Begin(long bytes, int rows, int layers, int decodeRows) { Budget = bytes; return 1; }
        public long Update(ulong token, long bytes) { Updates++; OnUpdate?.Invoke(bytes); Budget = bytes; return 0; }
        public long Trim(ulong token, long bytes) => 0;
        public long End(ulong token) => 0;
        public GgmlHostMoeExpertCacheUsage ReadUsage() => new(0, Budget, 0, 0, 0);
    }

    [Fact]
    public void ExpertShrinkFencesAndReleasesPhysicalChildrenBeforeRefundingEnvelopeCredit()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var envelope = budget.Reserve(DeviceCharges(400 * MiB));
        using var child = envelope.TryTake(DeviceCharges(300 * MiB))!; child.Commit();
        var native = new Native { OnUpdate = _ => {
            Assert.Equal(400 * MiB, envelope.OwnedCharges.Single().Bytes);
            Assert.Equal(300 * MiB, budget.Snapshot().Single(p => p.Pool == Device).Committed);
            child.Dispose();
            Assert.Equal(400 * MiB, envelope.OwnedCharges.Single().Bytes);
        }};
        using var cache = new GgmlHostMoeExpertCacheScope(400 * MiB, 512, 3, native, 3);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 200 * MiB, 200 * MiB,
            Charges(1000 * MiB, 1000 * MiB)));
        Assert.Equal(200 * MiB, cache.BudgetBytes);
        Assert.Equal(200 * MiB, envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == Device).Committed);
    }

    [Fact]
    public void NativeGrowthFailureRollsBackUnusedCreditAndKeepsEveryPhysicalChildOwned()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var envelope = budget.Reserve(DeviceCharges(600 * MiB));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600 * MiB, 512, 3, native, 3);
        Assert.True(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 400 * MiB, 200 * MiB, Charges(1000 * MiB, 1000 * MiB)));
        using var child = envelope.TryTake(DeviceCharges(300 * MiB))!; child.Commit();
        native.OnUpdate = _ => { Assert.Equal(600 * MiB, envelope.OwnedCharges.Single().Bytes); throw new IOException("native update fault"); };
        Assert.Throws<IOException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 600 * MiB, 200 * MiB,
            Charges(1000 * MiB, 1000 * MiB)));
        Assert.Equal(400 * MiB, cache.BudgetBytes);
        Assert.Equal(400 * MiB, envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(300 * MiB, envelope.LiveChildCharges.Single().Bytes);
    }

    [Fact]
    public void ExpertGrowthQuotaRefusalAndSubFloorShrinkNeverMutateNativeSettings()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var envelope = budget.Reserve(DeviceCharges(600 * MiB));
        var native = new Native();
        using var cache = new GgmlHostMoeExpertCacheScope(600 * MiB, 512, 3, native, 3);
        UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 400 * MiB, 200 * MiB, Charges(1000 * MiB, 1000 * MiB));
        using var competitor = budget.Reserve(DeviceCharges(500 * MiB));
        native.Updates = 0;
        Assert.False(UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 600 * MiB, 200 * MiB, Charges(1000 * MiB, 1000 * MiB)));
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 199 * MiB, 200 * MiB,
            Charges(1000 * MiB, 1000 * MiB)));
        Assert.Equal(0, native.Updates);
        Assert.Equal(400 * MiB, envelope.OwnedCharges.Single().Bytes);
    }

    [Fact]
    public void FailedPhysicalShrinkNeverRefundsTheStillOwnedChild()
    {
        var budget = new MemoryBudget(Charges(1000 * MiB, 1000 * MiB));
        using var envelope = budget.Reserve(DeviceCharges(400 * MiB));
        using var child = envelope.TryTake(DeviceCharges(300 * MiB))!; child.Commit();
        var native = new Native(); // Simulates a broken native release contract.
        using var cache = new GgmlHostMoeExpertCacheScope(400 * MiB, 512, 3, native, 3);
        Assert.Throws<InvalidOperationException>(() => UnifiedMemoryExpertCapacity.Resize(cache, envelope, Device, 200 * MiB, 200 * MiB,
            Charges(1000 * MiB, 1000 * MiB)));
        Assert.Equal(400 * MiB, envelope.OwnedCharges.Single().Bytes);
        Assert.Equal(300 * MiB, envelope.LiveChildCharges.Single().Bytes);
    }
}
