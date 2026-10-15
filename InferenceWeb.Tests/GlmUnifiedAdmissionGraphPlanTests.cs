// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Admission geometry only. These checks do not qualify native KV
/// growth, graph arithmetic, or a complete model/device.</summary>
public sealed class GlmUnifiedAdmissionGraphPlanTests
{
    private static GlmUnifiedMemoryGeometry Geometry() => new()
    {
        Layers = 45, DeclaredContextTokens = 32768, RecurrentLayers = 34,
        Hidden = 4096, Heads = 64, Vocab = 154880, KvRow = 512, IndexerDim = 128,
        KdaDim = 128, ConvKernel = 4, HyperConnections = 4, ExpertFf = 2048, UsedExperts = 8
    };

    [Fact]
    public void LongSiblingOnlyWidensTheJointDecodeGraphAndNotTheNewSerialPrefill()
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 128, 0, [32000], 2);
        Assert.Equal(256, plan.SerialContext);
        Assert.Equal(32256, plan.JointContext);
        Assert.Equal(128, plan.PrefillRows);
        Assert.Equal(2, plan.JointRows);
        Assert.Equal(geometry.GraphPeak(256, 128, false), plan.PrefillPeak);
        Assert.Equal(geometry.GraphPeak(32256, 2, true), plan.JointPeak);
        var impossibleCrossShape = geometry.GraphPeak(32256, 128, false);
        Assert.True(plan.MaximumPeak.Host < impossibleCrossShape.Host);
        Assert.True(plan.MaximumPeak.Device < impossibleCrossShape.Device);
    }

    [Fact]
    public void ActualJointOwnerCanReserveTheDispatchedShapesWithoutAnUndispatchedCrossProduct()
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 128, 0, [32000], 2);
        var target = plan.MaximumPeak;
        var budget = new MemoryBudget([new("host", target.Host), new("device", target.Device)]);
        using var joint = budget.Reserve([new("host", 1024), new("device", 2048)]);
        using var oldGraph = joint.TryTake([new("host", 512), new("device", 1024)])!;
        oldGraph.Commit();
        var before = budget.Snapshot();
        var oldCrossProduct = geometry.GraphPeak(plan.JointContext, plan.PrefillRows, false);
        Assert.False(joint.TryResizeTotal(Charges(oldCrossProduct)));
        Assert.Equal(before.ToArray(), budget.Snapshot().ToArray());
        Assert.True(joint.TryResizeTotal(Charges(target)));
        Assert.Equal(1024, joint.LiveChildCharges.Single(c => c.Pool == "device").Bytes);
        Assert.Equal(512, joint.LiveChildCharges.Single(c => c.Pool == "host").Bytes);
        Assert.Equal(target.Device, joint.OwnedCharges.Single(c => c.Pool == "device").Bytes);
    }

    [Theory]
    [InlineData(0, 256, 1)]
    [InlineData(17, 256, 17)]
    [InlineData(256, 256, 128)]
    [InlineData(257, 512, 128)]
    public void FreshSoloRequestKeepsItsOwnPaddedPromptContext(int prompt, int context, int rows)
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, prompt, 0, [], 2);
        Assert.Equal(context, plan.SerialContext);
        Assert.Equal(context, plan.JointContext);
        Assert.Equal(rows, plan.PrefillRows);
        Assert.Equal(1, plan.JointRows);
        Assert.Equal(geometry.GraphPeak(context, rows, false), plan.MaximumPeak);
    }

    [Fact]
    public void ComputedBoundaryAndEveryParticipatingHeadReserveTheActualNextGraph()
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 256, 256, [511, 1024], 2);
        Assert.Equal(512, plan.SerialContext); // One new row crosses the 256-token block.
        Assert.Equal(1280, plan.JointContext); // Third slot's next position is 1025.
        Assert.Equal(1, plan.PrefillRows);
        Assert.Equal(3, plan.JointRows);
        Assert.Equal(geometry.GraphPeak(1280, 3, true), plan.JointPeak);
    }

    [Fact]
    public void PaddingCannotExceedTheUnchangedLogicalContext()
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32767, 128, 32766, 32766, [32766], 2);
        Assert.Equal(32767, plan.SerialContext);
        Assert.Equal(32767, plan.JointContext);
        // Graph planning alone must not reduce the native full-slot estimate.
        Assert.True(geometry.SlotBytes(32767) > geometry.SlotBytes(256));
    }

    [Theory]
    [InlineData(17, 0, 17, 256)]
    [InlineData(354, 0, 354, 512)]
    [InlineData(1340, 0, 512, 1536)]
    [InlineData(1340, 1024, 316, 1536)]
    public void WidePrefillUsesOnlyTheActualPromptTailAndKeepsLongSiblingsOutOfItsShape(
        int prompt, int computed, int rows, int context)
    {
        var geometry = Geometry();
        var plan = GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 512, prompt, computed, [32000, 16384], 2);
        Assert.Equal(rows, plan.PrefillRows);
        Assert.Equal(context, plan.SerialContext);
        Assert.Equal(32256, plan.JointContext);
        Assert.Equal(3, plan.JointRows);
        Assert.Equal(geometry.GraphPeak(context, rows, false), plan.PrefillPeak);
        Assert.Equal(geometry.GraphPeak(32256, 3, true), plan.JointPeak);
        Assert.True(plan.PrefillPeak.Device < geometry.GraphPeak(32256, 512, false).Device);
    }

    [Fact]
    public void WideGraphForecastKeepsReplacementGenerationsAndCheckedOverflow()
    {
        var geometry = Geometry();
        var single = geometry.GraphPeak(1536, 512, false, 1);
        var replacement = geometry.GraphPeak(1536, 512, false, 2);
        Assert.Equal(2 * single.Host, replacement.Host);
        Assert.Equal(2 * single.Device, replacement.Device);
        var overflow = geometry with { Layers = int.MaxValue, RecurrentLayers = 0,
            DeclaredContextTokens = int.MaxValue };
        Assert.Throws<OverflowException>(() => overflow.Workspace(int.MaxValue, 512));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(32768)]
    public void InvalidLiveHeadsAreNotClampedIntoALegalGraph(int head)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedAdmissionGraphPlan.Create(
            Geometry(), 32768, 128, 17, 0, [head], 2));
    }

    [Fact]
    public void InvalidRequestGeometryIsRejectedBeforeForecasting()
    {
        var geometry = Geometry();
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 1025, 17, 0, [], 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 32769, 0, [], 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 17, 32768, [], 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => GlmUnifiedAdmissionGraphPlan.Create(geometry, 32768, 128, 17, 0, new int[8], 2));
    }

    private static MemoryCharge[] Charges(InferenceMemoryBytes bytes) => [new("host", bytes.Host), new("device", bytes.Device)];
}
