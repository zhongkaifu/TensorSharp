// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlHostMoeFilePipelineScopeTests
{
    private sealed class Native : IHostMoeFilePipelineScopeNative
    {
        internal int Begins, Ends;
        internal int ReaderVersion = 2;
        internal bool FailBegin, FailEnd, Active, Enabled;
        public int ParallelReaderVersion() => ReaderVersion;
        public ulong Begin(bool enabled, int minimumRows, long minimumPayloadBytes, long maximumRingBytes)
        {
            Begins++;
            if (FailBegin) throw new InvalidOperationException("begin");
            Assert.False(Active); Active = true; Enabled = enabled;
            return 17;
        }
        public void End(ulong token)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Ends++;
            if (FailEnd) throw new InvalidOperationException("end");
            Active = false;
        }
    }
    [Theory]
    [InlineData(32, 1, 128)]
    [InlineData(1048577, 1, 128)]
    [InlineData(64, 0, 128)]
    [InlineData(64, -1, 128)]
    [InlineData(64, 1, 127)]
    [InlineData(64, 1, 129)]
    [InlineData(64, 1, 16777344)]
    public void InvalidPolicyDoesNotEnterNative(int rows, long payload, long ring)
    {
        var native = new Native();
        Assert.Throws<ArgumentOutOfRangeException>(() => new GgmlHostMoeFilePipelineScope(true, rows, payload, ring, native));
        Assert.Equal(0, native.Begins);
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExplicitPolicyOwnsAndDisposesOnce(bool enabled)
    {
        var native = new Native();
        var scope = new GgmlHostMoeFilePipelineScope(enabled, 64, 32L << 20, 16L << 20, native);
        Assert.Equal(enabled, scope.Enabled); Assert.Equal(enabled, native.Enabled);
        Assert.Equal(64, scope.MinimumRows); Assert.Equal(32L << 20, scope.MinimumPayloadBytes);
        Assert.Equal(16L << 20, scope.MaximumRingBytes);
        scope.Dispose(); scope.Dispose();
        Assert.False(native.Active); Assert.Equal(1, native.Ends);
    }
    [Fact]
    public void OlderSequentialReaderCannotBePromotedAndHasNoOwnedCleanup()
    {
        foreach (int version in new[] { 0, 1 })
        {
            var native = new Native { ReaderVersion = version };
            Assert.Throws<NotSupportedException>(() => new GgmlHostMoeFilePipelineScope(true, 64, 1, 128, native));
            Assert.Equal(0, native.Begins); Assert.Equal(0, native.Ends); Assert.False(native.Active);
            using var disabled = new GgmlHostMoeFilePipelineScope(false, 64, 1, 128, native);
            Assert.False(disabled.Enabled); Assert.Equal(1, native.Begins);
        }
    }

    [Fact]
    public void FailedBeginHasNoOwnerToDispose()
    {
        var native = new Native { FailBegin = true };
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeFilePipelineScope(true, 64, 1, 128, native));
        Assert.False(native.Active); Assert.Equal(0, native.Ends);
    }
    [Fact]
    public void FailedDisposeRetainsTokenForRetry()
    {
        var native = new Native { FailEnd = true };
        var scope = new GgmlHostMoeFilePipelineScope(false, 64, 32L << 20, 16L << 20, native);
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.True(native.Active);
        native.FailEnd = false; scope.Dispose(); scope.Dispose();
        Assert.False(native.Active); Assert.Equal(2, native.Ends);
    }
}
