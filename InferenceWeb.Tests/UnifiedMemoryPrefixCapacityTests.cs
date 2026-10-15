using TensorSharp.Memory;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryPrefixCapacityTests
{
    [Theory]
    [InlineData(-1)]
    [InlineData(9)]
    [InlineData(int.MaxValue)]
    public void RequestedRetainedCopiesStayWithinTheSupportedBound(int requested)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryPrefixCapacity.Select(requested, 100, 8000));
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(-1L)]
    [InlineData(long.MinValue)]
    public void CopyGeometryMustBePositive(long bytes)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryPrefixCapacity.Select(4, bytes, 8000));
    }

    [Theory]
    [InlineData(-1L)]
    [InlineData(long.MinValue)]
    public void EffectiveDeviceCeilingCannotBeNegative(long ceiling)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => UnifiedMemoryPrefixCapacity.Select(4, 100, ceiling));
    }

    [Fact]
    public void DisabledPublicRetentionReservesNothing()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(0, 0), UnifiedMemoryPrefixCapacity.Select(0, 100, 8000));
    }

    [Fact]
    public void ZeroDeviceCeilingCannotPlaceAnyCopiedOwner()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(0, 0), UnifiedMemoryPrefixCapacity.Select(4, 100, 0));
    }

    [Theory]
    [InlineData(799L)]
    [InlineData(800L)]
    [InlineData(1599L)]
    public void OneFittingOwnerCannotEnableCaptureBeforeEviction(long ceiling)
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(0, 0), UnifiedMemoryPrefixCapacity.Select(4, 100, ceiling));
    }

    [Fact]
    public void FirstRetainedCopyProtectsOnlyItsOwnMandatoryPlacement()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(1, 100), UnifiedMemoryPrefixCapacity.Select(4, 100, 1600));
    }

    [Fact]
    public void RequestedRetainedCountDoesNotMakeOptionalReplacementAnExecutionFloor()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(4, 400), UnifiedMemoryPrefixCapacity.Select(4, 100, 4000));
    }

    [Fact]
    public void ExcessDeviceCapacityDoesNotExceedTheRequestedRetainedCount()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(8, 800), UnifiedMemoryPrefixCapacity.Select(8, 100, 16000));
    }

    [Theory]
    [InlineData(100L, 4, 400L)]
    [InlineData(300L, 2, 600L)]
    [InlineData(500L, 1, 500L)]
    [InlineData(600L, 0, 0L)]
    public void HeterogeneousCopyGeometryChangesPlacementUnderTheSameCeiling(long bytes, int retained, long reserve)
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(retained, reserve), UnifiedMemoryPrefixCapacity.Select(4, bytes, 8000));
    }

    [Fact]
    public void ImpossibleHugeCopiesDoNotOverflowBeforeCapacityIsBounded()
    {
        Assert.Equal(new UnifiedMemoryPrefixCapacity(0, 0),
            UnifiedMemoryPrefixCapacity.Select(8, long.MaxValue, long.MaxValue));
        Assert.Equal(new UnifiedMemoryPrefixCapacity(0, 0),
            UnifiedMemoryPrefixCapacity.Select(8, long.MaxValue / 9 + 1, long.MaxValue));
    }

    [Fact]
    public void LargeFittingCopiesBoundCountsBeforeMultiplicationAndLeaveReplacementOptional()
    {
        long bytes = long.MaxValue / 16;
        var selected = UnifiedMemoryPrefixCapacity.Select(8, bytes, long.MaxValue);
        Assert.Equal(1, selected.RetainedCopies);
        Assert.Equal(bytes, selected.PlacementReserveBytes);
        Assert.InRange(selected.PlacementReserveBytes, 1, long.MaxValue / 8);
        Assert.InRange(checked(selected.PlacementReserveBytes + bytes), 1, long.MaxValue / 8);
    }

    [Fact]
    public void OptionalReplacementMayMissWithoutReleasingRetainedOrMandatoryOwners()
    {
        var selected = UnifiedMemoryPrefixCapacity.Select(4, 100, 1600);
        Assert.Equal(1, selected.RetainedCopies); // The count allowance can hold two copies.
        var budget = new MemoryBudget([new("device", 1600)]);
        using var execution = budget.Reserve([new("device", 1500)]); execution.Commit();
        using var retained = budget.Reserve([new("device", selected.PlacementReserveBytes)]); retained.Commit();
        var before = budget.Snapshot().ToArray();
        Assert.Null(OwnedAllocationScope.TryReserveOptional(budget, [new("device", 100)], [new("device", 1600)]));
        Assert.Equal(before, budget.Snapshot().ToArray());
        Assert.Equal(100, Assert.Single(retained.OwnedCharges).Bytes);
        Assert.Equal(1500, Assert.Single(execution.OwnedCharges).Bytes);
    }
}
