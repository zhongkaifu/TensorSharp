using InferenceEngine;

namespace InferenceWeb.Tests;

public class MediaHelperTests
{
    private static readonly object EnvLock = new();

    [Fact]
    public void SelectEvenlySpacedIndicesReturnsAllIndicesWhenAlreadyUnderLimit()
    {
        var indices = MediaHelper.SelectEvenlySpacedIndices(count: 3, maxCount: 4);

        Assert.Equal(new[] { 0, 1, 2 }, indices);
    }

    [Fact]
    public void SelectEvenlySpacedIndicesIncludesEndpointsWhenDownsampling()
    {
        var indices = MediaHelper.SelectEvenlySpacedIndices(count: 8, maxCount: 4);

        Assert.Equal(4, indices.Count);
        Assert.Equal(0, indices[0]);
        Assert.Equal(7, indices[^1]);
        Assert.Equal(new[] { 0, 2, 5, 7 }, indices);
    }

    [Fact]
    public void SelectEvenlySpacedIndicesUsesMiddleFrameWhenOnlyOneIsRequested()
    {
        var indices = MediaHelper.SelectEvenlySpacedIndices(count: 9, maxCount: 1);

        Assert.Equal(new[] { 4 }, indices);
    }

    [Fact]
    public void GetConfiguredMaxVideoFramesFallsBackToDefaultWhenUnset()
    {
        lock (EnvLock)
        {
            string? oldValue = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
            try
            {
                Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", null);

                Assert.Equal(MediaHelper.DefaultVideoMaxFrames, MediaHelper.GetConfiguredMaxVideoFrames());
            }
            finally
            {
                Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", oldValue);
            }
        }
    }

    [Fact]
    public void GetConfiguredMaxVideoFramesUsesPositiveEnvironmentOverride()
    {
        lock (EnvLock)
        {
            string? oldValue = Environment.GetEnvironmentVariable("VIDEO_MAX_FRAMES");
            try
            {
                Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", "6");

                Assert.Equal(6, MediaHelper.GetConfiguredMaxVideoFrames());
            }
            finally
            {
                Environment.SetEnvironmentVariable("VIDEO_MAX_FRAMES", oldValue);
            }
        }
    }
}
