// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>Source-backed host admission/frontier policies, not a GPU/proposer qualification.</summary>
public sealed class NinferDraftHostPolicyTests
{
    private const string Policy = "original-dflash2:module-sha:payload-sha:k7";
    private static byte[] Pending() { var bytes = new byte[51200]; BinaryPrimitives.WriteUInt16LittleEndian(bytes, 0x8000); BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2), 0x3f81); return bytes; }

    [Fact]
    public void ZeroAcceptedDraftsAndZeroExtentUseDifferentOriginalEncodingPolicies()
    {
        // Source step.cu: extent>0,nAccepted0 still commits one anchor and
        // encodes the full physical window. Only extent0 carries raw taps.
        Assert.False(Q35CudaEngine.NinferDraftCarriesAnchor(8, 1));
        Assert.False(Q35CudaEngine.NinferDraftCarriesAnchor(7, 1));
        Assert.True(Q35CudaEngine.NinferDraftCarriesAnchor(1, 1));
        Assert.False(Q35CudaEngine.NinferDraftCarriesAnchor(1, 0));
        Assert.False(Q35CudaEngine.NinferDraftCarriesAnchor(8, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.NinferDraftCarriesAnchor(0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.NinferDraftCarriesAnchor(8, 9));
    }

    [Theory]
    [InlineData(4096, 0, 2048, 2048)]
    [InlineData(4096, 4096, 4096, 0)]
    [InlineData(1024, 24, 24, 1000)]
    [InlineData(7, 0, 0, 7)]
    [InlineData(0, 0, 0, 0)]
    public void PrefillTailHonorsWholeSpanTapStartAnd2048PositionCapacity(int real, int tapped, int first, int rows)
        => Assert.Equal((first, rows), Q35CudaEngine.NinferDraftPrefillTail(real, tapped));

    [Theory]
    [InlineData(7, 4, 28)]
    [InlineData(4, 8, 32)]
    [InlineData(1, 8, 8)]
    [InlineData(7, 1, 7)]
    public void FullHeadIsNeverSilentlySplitAcrossDifferentOriginalMmaRoutes(int draft, int lanes, int rows)
        => Assert.Equal(rows, Q35CudaEngine.ValidateNinferDraftHeadShape(draft, lanes));

    [Theory]
    [InlineData(7, 5)]
    [InlineData(7, 8)]
    [InlineData(5, 7)]
    public void ProposalRowsAbove33RequireTheMissingOriginalW8Route(int draft, int lanes)
        => Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateNinferDraftHeadShape(draft, lanes));

    [Fact]
    public void NormalAndCarriedFrontiersHaveDistinctOwnedCheckpointMetadata()
    {
        var complete = new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 10001, 10001, 7953, ReadOnlySpan<byte>.Empty);
        Q35CudaEngine.ValidateNinferDraftSnapshot(complete, 10001, Policy); Assert.True(complete.Pending.IsEmpty);
        byte[] pending = Pending();
        var carried = new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 10001, 10000, 7952, pending);
        pending[0] = 7; Assert.Equal(0, carried.Pending.Span[0]);
        Assert.Equal(0x8000, BinaryPrimitives.ReadUInt16LittleEndian(carried.Pending.Span));
        Assert.Equal(0x3f81, BinaryPrimitives.ReadUInt16LittleEndian(carried.Pending.Span[2..]));
        Q35CudaEngine.ValidateNinferDraftSnapshot(carried, 10001, Policy);
    }

    [Fact]
    public void MissingOrForeignRestoreMetadataCannotPretendToBeAnEmptyNewSlot()
    {
        var snapshot = new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 128, 127, 0, Pending());
        Assert.Throws<ArgumentNullException>(() => Q35CudaEngine.ValidateNinferDraftSnapshot(null!, 128, Policy));
        Assert.Throws<InvalidOperationException>(() => Q35CudaEngine.ValidateNinferDraftSnapshot(snapshot, 127, Policy));
        Assert.Throws<InvalidOperationException>(() => Q35CudaEngine.ValidateNinferDraftSnapshot(snapshot, 128, "different-module-or-weights"));
        Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 128, 127, 0, ReadOnlySpan<byte>.Empty));
        Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 129, 127, 0, Pending()));
        Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 128, 129, 0, ReadOnlySpan<byte>.Empty));
    }

    [Theory]
    [InlineData(51199)]
    [InlineData(51201)]
    public void TruncatedOrOversizedPendingColumnIsRefused(int bytes)
        => Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 1, 0, 0, new byte[bytes]));

    [Theory]
    [InlineData(0x7f80)]
    [InlineData(0xff80)]
    [InlineData(0x7fc1)]
    public void NonfinitePendingBf16DoesNotEnterRestoredDraftState(int bits)
    {
        byte[] pending = Pending(); BinaryPrimitives.WriteUInt16LittleEndian(pending.AsSpan(51198), (ushort)bits);
        Assert.Throws<InvalidDataException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 1, 0, 0, pending));
    }

    [Fact]
    public void EarliestAvailablePositionCannotInventHistoryOutsideThePhysicalRing()
    {
        Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 4096, 4096, 0, ReadOnlySpan<byte>.Empty));
        Assert.Throws<InvalidOperationException>(() => new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 7, 7, 8, ReadOnlySpan<byte>.Empty));
        var empty = new Q35CudaEngine.NinferDraftSlotSnapshot(Policy, 0, 0, 0, ReadOnlySpan<byte>.Empty);
        Q35CudaEngine.ValidateNinferDraftSnapshot(empty, 0, Policy);
    }
}
