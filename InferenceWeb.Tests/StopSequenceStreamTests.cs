// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Server;

namespace InferenceWeb.Tests;

/// <summary>The chat pipeline's stop-sequence stream: a matched stop sequence is never sent, and text that
/// could begin one is held back only until a later piece (or the end of the turn) settles it.</summary>
public sealed class StopSequenceStreamTests
{
    [Fact]
    public void APrefixIsHeldBack_UntilTheNextPieceSettlesIt()
    {
        var stream = StopSequenceStream.For(new[] { "world peace" })!;
        Assert.Equal("Hello ", stream.Append("Hello wor"));
        Assert.Equal("world!", stream.Append("ld!"));
        Assert.False(stream.Matched);
        Assert.Equal("Hello world!", stream.SentText);
    }

    [Fact]
    public void AStopSplitAcrossPieces_IsNotSent_NorAnythingAfterIt()
    {
        var stream = StopSequenceStream.For(new[] { "</answer>" })!;
        Assert.Equal("42", stream.Append("42</"));
        Assert.Equal("", stream.Append("answ"));
        Assert.Equal("", stream.Append("er> and more"));
        Assert.True(stream.Matched);
        Assert.Equal("", stream.Append("later"));
        Assert.Equal("", stream.Flush());
        Assert.Equal("42", stream.SentText);
    }

    [Fact]
    public void TheEarliestMatch_EndsTheText_WhicheverStopItIs()
    {
        var stream = StopSequenceStream.For(new[] { "c", "ab" })!;
        Assert.Equal("x", stream.Append("xabc"));
        Assert.True(stream.Matched);
    }

    [Fact]
    public void HeldBackText_IsSentByFlush_WhenTheTurnEndsAnotherWay()
    {
        var stream = StopSequenceStream.For(new[] { "\n\nUser:" })!;
        Assert.Equal("Done.", stream.Append("Done.\n\n"));
        Assert.Equal("\n\n", stream.Flush());
        Assert.Equal("Done.\n\n", stream.SentText);
    }

    [Fact]
    public void NoUsableStop_GivesNoStream()
    {
        Assert.Null(StopSequenceStream.For(null));
        Assert.Null(StopSequenceStream.For(new[] { "", null! }));
    }
}
