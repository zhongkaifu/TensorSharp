using System.Runtime.CompilerServices;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>Historical position bookkeeping only; media encoder/model quality is separate.</summary>
public class Qwen4ExpMtpPositionTests
{
    [Fact]
    public void DelayedCatchUp_UsesBothMediaChunksAndTheirHistoricalTextGap()
    {
        var ranges = new List<Qwen4ExpModel.MtpPositionRange>
        {
            Text(0, 2, 0),
            Media(2, [2, 10, 20, 2, 11, 20]),
            Media(4, [3, 12, 21, 4, 12, 21]),
            Text(6, 3, 5),
        };
        var actual = Qwen4ExpModel.ResolveMtpPositions(ranges, 1, 7, fallbackGap: -100);
        Assert.Equal(new[] { 1,1,1, 2,10,20, 2,11,20, 3,12,21, 4,12,21, 5,5,5, 6,6,6 }, actual.MultiAxis);
        Assert.Equal(1, actual.RopePosition);
        var oldText = Qwen4ExpModel.ResolveMtpPositions(ranges, 0, 2, fallbackGap: 90);
        Assert.Null(oldText.MultiAxis);
        Assert.Equal(0, oldText.RopePosition);
    }

    [Fact]
    public void ReplayedTail_TruncatesOldMediaRangeAndDoesNotReuseAbandonedCoordinates()
    {
        var original = Media(2, [2,10,20, 3,11,21, 4,12,22, 5,13,23]);
        int[] before = (int[])original.MultiAxis.Clone();
        var ranges = new List<Qwen4ExpModel.MtpPositionRange> { Text(0, 2, 0), original, Text(6, 4, 6) };
        Qwen4ExpModel.PublishMtpPositions(ranges, Text(4, 3, 30));
        Assert.Equal(3, ranges.Count);
        Assert.Equal(2, original.Count);
        Assert.Equal(before, original.MultiAxis);
        var actual = Qwen4ExpModel.ResolveMtpPositions(ranges, 3, 4, fallbackGap: 0);
        Assert.Equal(new[] { 3,11,21, 30,30,30, 31,31,31, 32,32,32 }, actual.MultiAxis);
        Qwen4ExpModel.PublishMtpPositions(ranges, Text(0, 2, 50));
        Assert.Single(ranges);
        Assert.Equal(50, Qwen4ExpModel.ResolveMtpPositions(ranges, 0, 2, fallbackGap: 0).RopePosition);
    }

    [Fact]
    public void Overlap_UsesLatestRangeAndUnknownGapOnlyForUnrecordedTokens()
    {
        var ranges = new List<Qwen4ExpModel.MtpPositionRange> { Text(2, 4, 40), Text(3, 2, 80) };
        var actual = Qwen4ExpModel.ResolveMtpPositions(ranges, 1, 6, fallbackGap: -10);
        Assert.Equal(new[] { 11,11,11, 40,40,40, 80,80,80, 81,81,81, 43,43,43, 16,16,16 }, actual.MultiAxis);
    }

    [Fact]
    public void CapturedPositions_CopyCallerArrayAndReleaseOnlyTheirActualOwner()
    {
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        try
        {
            object a = new(), b = new();
            var map = new Dictionary<object, List<Qwen4ExpModel.MtpPositionRange>>(ReferenceEqualityComparer.Instance);
            Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_mtpPositions", map);
            int[] axes = [2,4,6, 3,5,7];
            Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_pendingMRoPEPositions", axes);
            var capture = (Qwen4ExpModel.MtpPositionRange)Qwen4ExpMtpMathTests.Invoke(model, "PrepareMtpPositions", [a, 2])!;
            axes[0] = 999;
            Qwen4ExpModel.PublishMtpPositions(map[a], capture);
            Assert.Equal(new[] { 2,4,6, 3,5,7 }, capture.MultiAxis);
            var other = (Qwen4ExpModel.MtpPositionRange)Qwen4ExpMtpMathTests.Invoke(model, "PrepareMtpPositions", [b, 2])!;
            Qwen4ExpModel.PublishMtpPositions(map[b], other);
            Qwen4ExpMtpMathTests.Invoke(model, "ReleaseMtpState", [a]);
            Assert.False(map.ContainsKey(a));
            Assert.Single(map[b]);
            Assert.Equal(999, map[b][0].MultiAxis[0]);
            Qwen4ExpMtpMathTests.Invoke(model, "PruneMtpPositions", [b, 1]);
            Assert.Single(map[b]);
            Qwen4ExpMtpMathTests.Invoke(model, "PruneMtpPositions", [b, 2]);
            Assert.Empty(map[b]);
        }
        finally { GC.SuppressFinalize(model); }
    }

    [Theory]
    [InlineData(-1, 1)]
    [InlineData(0, 0)]
    [InlineData(int.MaxValue, 1)]
    public void InvalidRequestBounds_DeclineBeforeReadingPositionRanges(int position, int count)
        => Assert.Throws<ArgumentOutOfRangeException>(() => Qwen4ExpModel.ResolveMtpPositions([], position, count, 0));

    [Fact]
    public void TimeCoordinatesAheadOfKv_PreserveSignedGapForFollowingText()
    {
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        try
        {
            var map = new Dictionary<object, List<Qwen4ExpModel.MtpPositionRange>>(ReferenceEqualityComparer.Instance);
            Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_mtpPositions", map);
            Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_pendingMRoPEPositions", new[] { 10,1,2, 11,2,2 });
            Qwen4ExpMtpMathTests.Invoke(model, "UpdateMropeGap", [0, 2]);
            Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), model, "_pendingMRoPEPositions", null!);
            Qwen4ExpMtpMathTests.Set(typeof(ModelBase), model, "_cacheSeqLen", 2);
            var text = (Qwen4ExpModel.MtpPositionRange)Qwen4ExpMtpMathTests.Invoke(model, "PrepareMtpPositions", [new object(), 2])!;
            Assert.Null(text.MultiAxis);
            Assert.Equal(12, text.RopePosition);
        }
        finally { GC.SuppressFinalize(model); }
    }

    [Fact]
    public void VideoClip_CatchUpReplaysBothTemporalPairsWithIncreasingTimeAndTheLabelTextBetween()
    {
        // A two-pair clip as the Qwen-VL injector lays it out: text (0..1), pair 1 at
        // running position 2 on a 2x2 merged grid, one label token, pair 2 at 5, then
        // text resuming at 8. The draft's delayed catch-up must see exactly those
        // coordinates, with the second pair strictly later in time than the first.
        var pair1 = Media(2, [2,2,2, 2,2,3, 2,3,2, 2,3,3]);
        var pair2 = Media(7, [5,5,5, 5,5,6, 5,6,5, 5,6,6]);
        var ranges = new List<Qwen4ExpModel.MtpPositionRange> { Text(0, 2, 0), pair1, Text(6, 1, 4), pair2, Text(11, 2, 7) };
        var actual = Qwen4ExpModel.ResolveMtpPositions(ranges, 1, 12, fallbackGap: -100);
        Assert.Equal(new[] { 1,1,1, 2,2,2, 2,2,3, 2,3,2, 2,3,3, 4,4,4, 5,5,5, 5,5,6, 5,6,5, 5,6,6, 7,7,7, 8,8,8 }, actual.MultiAxis);
        Assert.Equal(1, actual.RopePosition);
        Assert.True(actual.MultiAxis![3 * 6] > actual.MultiAxis[3 * 1]);
        // Catching up over only the second pair still uses its recorded coordinates.
        var late = Qwen4ExpModel.ResolveMtpPositions(ranges, 7, 4, fallbackGap: 0);
        Assert.Equal(pair2.MultiAxis, late.MultiAxis);
        Assert.Equal(5, late.RopePosition);
    }

    [Fact]
    public void UniformHistoricalCoordinates_UseScalarWithoutLatestGapReinterpretation()
    {
        var ranges = new List<Qwen4ExpModel.MtpPositionRange> { Media(7, [50,50,50, 51,51,51]), Text(9, 2, 52) };
        var actual = Qwen4ExpModel.ResolveMtpPositions(ranges, 7, 4, fallbackGap: 6);
        Assert.Null(actual.MultiAxis);
        Assert.Equal(50, actual.RopePosition);
    }

    private static Qwen4ExpModel.MtpPositionRange Text(int position, int count, int rope)
        => new() { Position = position, Count = count, RopePosition = rope };
    private static Qwen4ExpModel.MtpPositionRange Media(int position, int[] axes)
        => new() { Position = position, Count = axes.Length / 3, RopePosition = axes[0], MultiAxis = axes };
}
