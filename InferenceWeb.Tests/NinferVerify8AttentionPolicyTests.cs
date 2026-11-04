// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferVerify8AttentionPolicyTests
{
    private const long ElementsBytes = 8L * 24 * 256 * 4;
    private static readonly IntPtr Q = new(0x100000), Gate = new(0x200000), Blocks = new(0x300000),
        Position = new(0x400000), Output = new(0x500000), Workspace = new(0x600000);

    private static CudaNinferPrefillAttentionModule.Layout Admit(IntPtr? q = null, IntPtr? gate = null,
        IntPtr? blocks = null, IntPtr? position = null, IntPtr? output = null, IntPtr? workspace = null,
        long? workspaceBytes = null, float scale = .0625f)
        => CudaNinferPrefillAttentionModule.ValidateVerify8Buffers(q ?? Q, gate ?? Gate, blocks ?? Blocks, 160,
            0, 524288, position ?? Position, 40960, scale, output ?? Output, workspace ?? Workspace,
            workspaceBytes ?? CudaNinferPrefillAttentionModule.WorkspaceVerify8());

    [Fact]
    public void FixedCapacityDoesNotCaptureShortHistoryOrItsPageCount()
    {
        var capacity = CudaNinferPrefillAttentionModule.PlanVerify8(40960, 160);
        foreach (int visible in new[] { 8, 9, 63, 64, 65, 255, 256, 257, 40952, 40959, 40960 })
        {
            int minimumBlocks = (visible + 255) / 256;
            Assert.Equal(capacity, CudaNinferPrefillAttentionModule.PlanVerify8(visible, minimumBlocks));
            Assert.Equal(capacity, CudaNinferPrefillAttentionModule.PlanVerify8(visible, 161));
        }
        Assert.Equal(640, capacity.Pages);
        Assert.Equal(4, capacity.Splits);
        Assert.Equal(992256, capacity.Bytes);
        Assert.Equal(capacity.Bytes, CudaNinferPrefillAttentionModule.WorkspaceVerify8());
        // A larger declared model/table capacity does not enlarge visible history
        // or become a captured core/page-count argument.
        Assert.Equal(capacity, CudaNinferPrefillAttentionModule.PlanVerify8(202, 257));
        Assert.Equal(capacity, CudaNinferPrefillAttentionModule.PlanVerify8(40960, 1025));
    }

    [Fact]
    public void OriginalT8ShapesFitWithoutAFullHistoryCopy()
    {
        var p = Admit();
        // Independent logical tensor sizes for8×24×256 and four splits.
        long[] sizes = { 98304, 98304, 786432, 3072, 3072, 32, 4, 2560 };
        long[] offsets = { p.Query, p.Output, p.PartialAcc, p.PartialM, p.PartialL, p.Planes, p.Positions, p.Table, p.Bytes };
        Assert.All(offsets, x => Assert.Equal(0, x % 256));
        for (int i = 0; i < sizes.Length; i++)
            Assert.InRange(offsets[i] + sizes[i], offsets[i], offsets[i + 1]);
        Assert.True(p.Bytes < 1L << 20);
        Assert.Equal(40960, CudaNinferPrefillAttentionModule.Verify8Context);
    }

    [Theory]
    [InlineData(8, 1, true, true, 8, true)]
    [InlineData(8, 1, true, true, 40960, true)]
    [InlineData(7, 1, true, true, 40960, false)]
    [InlineData(9, 1, true, true, 40960, false)]
    [InlineData(8, 0, true, true, 40960, false)]
    [InlineData(8, 2, true, true, 40960, false)]
    [InlineData(8, 1, false, true, 40960, false)]
    [InlineData(8, 1, true, false, 40960, false)]
    [InlineData(8, 1, true, true, 7, false)]
    [InlineData(8, 1, true, true, 40961, false)]
    public void OnlyFullValidSingleLaneVerify8CanBeSelected(int rows, int sequences, bool verify, bool allValid,
        int visible, bool expected)
        => Assert.Equal(expected, CudaNinferPrefillAttentionModule.SupportsVerify8(rows, sequences, verify, allValid, visible));

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(7, 1)]
    [InlineData(40961, 161)]
    [InlineData(int.MaxValue, 161)]
    [InlineData(8, -1)]
    [InlineData(8, 0)]
    [InlineData(257, 1)]
    [InlineData(40960, 159)]
    [InlineData(8, 1026)]
    [InlineData(8, int.MaxValue)]
    public void InvalidHistoryAndPointerTableCapacityFailClosed(int visible, int blocks)
        => Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.PlanVerify8(visible, blocks));

    [Fact]
    public void OrdinaryPrefillAdmissionRemainsUnchanged()
    {
        Assert.False(CudaNinferPrefillAttentionModule.Supports(8, 1, false));
        Assert.False(CudaNinferPrefillAttentionModule.Supports(8, 1, true));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.Plan(8, 40960, 160, 170));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.SplitCount(8, 170));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaNinferPrefillAttentionModule.WorkspaceCapacity(8, 40960));
        Assert.True(CudaNinferPrefillAttentionModule.Supports(64, 1, false));
    }

    [Fact]
    public void DevicePositionCannotAliasAnyOtherLiveSpanIncludingUnusedPackPosition()
    {
        var p = Admit();
        foreach (IntPtr alias in new[] { Q, Q + (nint)(ElementsBytes - 4), Gate, Blocks, Blocks + 1276,
                     Output, Workspace, Workspace + (nint)p.Positions, Workspace + (nint)(p.Bytes - 4) })
            Assert.Throws<ArgumentException>(() => Admit(position: alias));
        // Adjacency is legal; neither last-byte nor first-byte overlap is hidden.
        Admit(position: Q + (nint)ElementsBytes);
        Admit(position: Blocks + 1280);
        Admit(position: Workspace + (nint)p.Bytes);
    }

    [Fact]
    public void OutputAndWorkspaceCannotOverwriteInputsMetadataOrEachOther()
    {
        foreach (IntPtr alias in new[] { Q, Gate, Blocks, Position, Workspace })
            Assert.Throws<ArgumentException>(() => Admit(output: alias));
        foreach (IntPtr alias in new[] { Q, Gate, Blocks, Position, Output })
            Assert.Throws<ArgumentException>(() => Admit(workspace: alias));
        long larger = CudaNinferPrefillAttentionModule.WorkspaceVerify8() + 256;
        Assert.Throws<ArgumentException>(() => Admit(position: Workspace + (nint)(larger - 4), workspaceBytes: larger));
    }

    [Fact]
    public void NullUnalignedOverflowingAndShortBuffersFailBeforeCudaIsNeeded()
    {
        Assert.Throws<ArgumentException>(() => Admit(q: IntPtr.Zero));
        Assert.Throws<ArgumentException>(() => Admit(gate: Gate + 1));
        Assert.Throws<ArgumentException>(() => Admit(blocks: Blocks + 4));
        Assert.Throws<ArgumentException>(() => Admit(position: Position + 2));
        Assert.Throws<ArgumentException>(() => Admit(output: Output + 2));
        Assert.Throws<ArgumentException>(() => Admit(workspace: Workspace + 4));
        Assert.Throws<ArgumentException>(() => Admit(q: new IntPtr(long.MaxValue & ~3L)));
        Assert.Throws<ArgumentException>(() => Admit(workspaceBytes: CudaNinferPrefillAttentionModule.WorkspaceVerify8() - 1));
        Assert.Throws<ArgumentException>(() => Admit(workspaceBytes: long.MaxValue));
    }

    [Fact]
    public void AttentionScalingAndPlaneAdmissionRemainExact()
    {
        foreach (float scale in new[] { 0f, -.0625f, .125f, float.NaN, float.NegativeInfinity, float.PositiveInfinity })
            Assert.Throws<ArgumentOutOfRangeException>(() => Admit(scale: scale));
        Assert.Throws<ArgumentException>(() => CudaNinferPrefillAttentionModule.ValidateVerify8Buffers(Q, Gate, Blocks, 160,
            0, 0, Position, 40960, .0625f, Output, Workspace, CudaNinferPrefillAttentionModule.WorkspaceVerify8()));
    }
}
