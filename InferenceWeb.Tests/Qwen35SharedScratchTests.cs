// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

// Pure geometry and pointer arithmetic. These tests never create a CUDA context, and
// do not qualify model numerics or GPU launchability of the shared-storage opt-in.
public sealed class Qwen35SharedScratchTests
{
    [Fact]
    public void Qwen27B_FullMixedCapacity_SavesExactlyTheTenSeparateAllocations()
    {
        var plan = Q35CudaEngine.PlanSharedScratch(4160, 5120, 6144, 10240, 48, 6144, 17408);
        Assert.Equal(596377600, plan.PoolBytes);
        Assert.Equal(1058037760, plan.LegacyBytes);
        Assert.Equal(461660160, plan.BytesSaved);
        Assert.Equal(10, plan.Spans.Count);
        Assert.Equal(10, plan.Spans.Select(s => s.Name).Distinct().Count());
        var xn = Assert.Single(plan.Spans, s => s.Name == "_xn");
        Assert.Equal(85196800, xn.Bytes);
        Assert.Equal(0, xn.Offset);
        Assert.Equal(new[] { "normalize", "attention", "gdn" }, plan.Spans.Select(s => s.Phase).Distinct());
        Assert.Equal(596377600, PhaseSize(plan, "attention"));
        Assert.Equal(376463360, PhaseSize(plan, "gdn"));
        Assert.Equal(85196800, PhaseSize(plan, "normalize"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(65)]
    [InlineData(4160)]
    public void RaggedAndZeroRows_KeepEveryLivePhaseDisjointAndWithinTheSlab(int rows)
    {
        // Odd widths require real padding rather than relying on model dimensions
        // happening to be multiples of 256 bytes.
        var widths = new Dictionary<string, int> {
            ["_xn"] = 5, ["_q"] = 7, ["_gate"] = 7, ["_attn"] = 7, ["_act"] = 17,
            ["_gdnQkv"] = 11, ["_gdnG"] = 3, ["_gdnBeta"] = 3, ["_gdnO"] = 13, ["_gdnY"] = 13
        };
        var plan = Q35CudaEngine.PlanSharedScratch(rows, 5, 7, 11, 3, 13, 17);
        foreach (var phase in plan.Spans.GroupBy(s => s.Phase))
        {
            long end = 0;
            foreach (var span in phase)
            {
                Assert.Equal((long)rows * widths[span.Name] * sizeof(float), span.Bytes);
                Assert.Equal(0, span.Offset % 256);
                Assert.Equal(0, span.CapacityBytes % 256);
                Assert.True(span.CapacityBytes >= Math.Max(256, span.Bytes));
                Assert.True(span.Offset >= end, $"{span.Name} overlaps another live buffer");
                end = checked(span.Offset + span.CapacityBytes);
                Assert.InRange(end, 256, plan.PoolBytes);
            }
        }
        Assert.Equal(plan.Spans.GroupBy(s => s.Phase).Max(p => p.Max(s => s.Offset + s.CapacityBytes)), plan.PoolBytes);
        Assert.Equal(plan.Spans.Sum(s => Math.Max(256, s.Bytes)), plan.LegacyBytes);
        Assert.True(plan.BytesSaved >= 0);
    }

    [Fact]
    public void ZeroWidthMembers_StillHaveAValidMinimumAllocationAndNoOverlap()
    {
        var plan = Q35CudaEngine.PlanSharedScratch(4160, 0, 0, 0, 0, 0, 0);
        Assert.Equal(1280, plan.PoolBytes);
        Assert.Equal(2560, plan.LegacyBytes);
        Assert.All(plan.Spans, s => { Assert.Equal(0, s.Bytes); Assert.Equal(256, s.CapacityBytes); });
    }

    [Fact]
    public void Mixed4160Rows_ThePrefillAndVerifySlicesFitEveryFullRowView()
    {
        const int prefill = 4096, verify = 8 * 8, rows = prefill + verify;
        var plan = Q35CudaEngine.PlanSharedScratch(rows, 5120, 6144, 10240, 48, 6144, 17408);
        foreach (var span in plan.Spans)
        {
            long rowBytes = span.Bytes / rows;
            long verifyStart = span.Offset + prefill * rowBytes;
            Assert.Equal(span.Offset + span.Bytes, verifyStart + verify * rowBytes);
            Assert.True(verifyStart >= span.Offset);
            Assert.True(verifyStart + verify * rowBytes <= span.Offset + span.CapacityBytes);
        }
    }

    [Fact]
    public void PointerViews_AreAlignedBoundedAndOverlapOnlyAcrossPhases()
    {
        var plan = Q35CudaEngine.PlanSharedScratch(65, 5, 7, 11, 3, 13, 17);
        var backing = new IntPtr(0x100000000);
        foreach (var span in plan.Spans)
        {
            long pointer = Q35CudaEngine.SharedScratchPointer(backing, plan.PoolBytes, span).ToInt64();
            Assert.Equal(backing.ToInt64() + span.Offset, pointer);
            Assert.Equal(0, pointer % 256);
            Assert.True(pointer + span.CapacityBytes <= backing.ToInt64() + plan.PoolBytes);
        }
        // Each phase intentionally starts at the same base, but attention's _act
        // must remain distinct from its q/gate/output, used in the same kernels.
        Assert.All(plan.Spans.GroupBy(s => s.Phase), phase => Assert.Equal(0, phase.First().Offset));
        var attention = plan.Spans.Where(s => s.Phase == "attention").ToArray();
        Assert.Equal(4, attention.Select(s => Q35CudaEngine.SharedScratchPointer(backing, plan.PoolBytes, s)).Distinct().Count());
    }

    [Fact]
    public void InvalidOrOverflowingGeometry_IsRejectedBeforeAnyPointerIsProduced()
    {
        for (int negative = 0; negative < 7; negative++)
        {
            int[] shape = { 1, 1, 1, 1, 1, 1, 1 };
            shape[negative] = -1;
            Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.PlanSharedScratch(
                shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6]));
        }
        // A single F32 span overflows, and a separate case overflows the phase
        // sum even though each individual multiplication fits in Int64.
        Assert.Throws<OverflowException>(() => Q35CudaEngine.PlanSharedScratch(int.MaxValue, int.MaxValue, 1, 1, 1, 1, 1));
        Assert.Throws<OverflowException>(() => Q35CudaEngine.PlanSharedScratch(int.MaxValue, 1, 536870912, 1, 1, 1, 1));
    }

    [Fact]
    public void InvalidSpansAndAddressOverflow_CannotEscapeTheOwnedAllocation()
    {
        var backing = new IntPtr(0x100000000);
        var valid = new Q35CudaEngine.ScratchLayoutSpan("test", "test", 0, 256, 256);
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.SharedScratchPointer(IntPtr.Zero, 256, valid));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.SharedScratchPointer(backing + 1, 256, valid));
        foreach (var invalid in new[] {
            valid with { Offset = -256 }, valid with { Offset = 1 }, valid with { Offset = 256 },
            valid with { Bytes = -1 }, valid with { Bytes = 257 }, valid with { CapacityBytes = 0 },
            valid with { CapacityBytes = 257 }
        })
            Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.SharedScratchPointer(backing, 256, invalid));
        Assert.Throws<ArgumentOutOfRangeException>(() => Q35CudaEngine.SharedScratchPointer(backing, -1, valid));
        var highAddress = new IntPtr(long.MaxValue & ~255L);
        Assert.Throws<OverflowException>(() => Q35CudaEngine.SharedScratchPointer(highAddress, 512, valid with { Offset = 256 }));
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("0", false)]
    [InlineData("true", false)]
    [InlineData("01", false)]
    [InlineData(" 1", false)]
    [InlineData("1", true)]
    public void Sharing_RequiresTheExactExplicitOptIn(string? value, bool expected)
        => Assert.Equal(expected, Q35CudaEngine.SharedScratchRequested(value));

    private static long PhaseSize(Q35CudaEngine.SharedScratchLayout plan, string phase)
        => plan.Spans.Where(s => s.Phase == phase).Max(s => s.Offset + s.CapacityBytes);
}
