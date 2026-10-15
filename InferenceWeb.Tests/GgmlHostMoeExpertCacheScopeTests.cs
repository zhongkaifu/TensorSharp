// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

public sealed class GgmlHostMoeExpertCacheScopeTests
{
    private sealed class Native : IHostMoeExpertCacheScopeNative
    {
        public int Begins, Updates, Trims, Ends, MaxDecodeRows;
        public bool FailBegin, FailUpdate, FailTrim, FailEnd, Active;
        public long Budget;
        public ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows)
        {
            Begins++;
            if (FailBegin) throw new InvalidOperationException("begin");
            Assert.False(Active);
            Active = true; Budget = bytes; MaxDecodeRows = maxDecodeRows;
            return 17;
        }
        public long Update(ulong token, long bytes)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Updates++;
            if (FailUpdate) throw new InvalidOperationException("update");
            Budget = bytes; return 11;
        }
        public long Trim(ulong token, long bytes)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Trims++;
            if (FailTrim) throw new InvalidOperationException("trim");
            return 13;
        }
        public long End(ulong token)
        {
            Assert.Equal(17UL, token); Assert.True(Active); Ends++;
            if (FailEnd) throw new InvalidOperationException("end");
            Active = false; return 19;
        }
        public GgmlHostMoeExpertCacheUsage ReadUsage()
        { Assert.True(Active); return new(14, Budget, 20, 30, 40); }
    }

    private sealed class GrowthNative : IHostMoeExpertCacheScopeNative
    {
        internal readonly Native Inner = new();
        internal bool Enabled;
        public ulong Begin(long bytes, int rows, int layers, int decodeRows) => Inner.Begin(bytes, rows, layers, decodeRows);
        public ulong Begin(long bytes, int rows, int layers, int decodeRows, bool enabled)
        { Enabled = enabled; return Begin(bytes, rows, layers, decodeRows); }
        public long Update(ulong token, long bytes) => Inner.Update(token, bytes);
        public long Trim(ulong token, long bytes) => Inner.Trim(token, bytes);
        public long End(ulong token) => Inner.End(token);
        public GgmlHostMoeExpertCacheUsage ReadUsage() => Inner.ReadUsage();
    }

    [Fact]
    public void LegacyNativeCannotSilentlyEnableGrowthOrCreateAnUnreachableOwner()
    {
        var native = new Native();
        Assert.Throws<NotSupportedException>(() => new GgmlHostMoeExpertCacheScope(100, 32, 48, native, 3, true));
        Assert.Equal(0, native.Begins); Assert.Equal(0, native.Ends); Assert.False(native.Active);
    }

    [Fact]
    public void GrowthIsOneAtomicBeginAndOrdinaryCallersStayDisabled()
    {
        var native = new GrowthNative();
        using (var scope = new GgmlHostMoeExpertCacheScope(100, 32, 48, native, 3, true))
        { Assert.True(scope.CapacityGrowthEnabled); Assert.True(native.Enabled); Assert.Equal(1, native.Inner.Begins); }
        Assert.False(native.Inner.Active); Assert.Equal(1, native.Inner.Ends);
        using var ordinary = new GgmlHostMoeExpertCacheScope(100, 32, 48, native, 3);
        Assert.False(ordinary.CapacityGrowthEnabled); Assert.False(native.Enabled);
    }

    [Theory]
    [InlineData(-1, 64, 48)]
    [InlineData(100, 0, 48)]
    [InlineData(100, 1025, 48)]
    [InlineData(100, 64, 0)]
    public void InvalidSettingsDoNotEnterNative(long bytes, int rows, int layers)
    {
        var native = new Native();
        Assert.Throws<ArgumentOutOfRangeException>(() => new GgmlHostMoeExpertCacheScope(bytes, rows, layers, native));
        Assert.Equal(0, native.Begins);
    }

    [Theory]
    [InlineData(129)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void WideScopedGlmPrefillPreservesTheAdmittedCapAndDecodeWidth(int rows)
    {
        var native = new Native();
        using (var scope = new GgmlHostMoeExpertCacheScope(100, rows, 3, native, 3))
        {
            Assert.Equal(rows, scope.MaxPrefillRows);
            Assert.Equal(3, scope.MaxDecodeRows);
            Assert.Equal(3, native.MaxDecodeRows);
            Assert.Equal(1, native.Begins);
        }
        Assert.Equal(1, native.Ends);
        Assert.False(native.Active);
    }

    [Fact]
    public void FailedBeginHasNoOwnedCleanup()
    {
        var native = new Native { FailBegin = true };
        Assert.Throws<InvalidOperationException>(() => new GgmlHostMoeExpertCacheScope(100, 64, 48, native));
        Assert.False(native.Active); Assert.Equal(0, native.Ends);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(9)]
    public void InvalidDecodePlanDoesNotEnterNative(int rows)
    {
        var native = new Native();
        Assert.Throws<ArgumentOutOfRangeException>(() => new GgmlHostMoeExpertCacheScope(100, 32, 48, native, rows));
        Assert.Equal(0, native.Begins);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(8)]
    public void DecodePlanIsIndependentOfPrefillLimit(int rows)
    {
        var native = new Native();
        using var scope = new GgmlHostMoeExpertCacheScope(100, 1, 48, native, rows);
        Assert.Equal(rows, scope.MaxDecodeRows); Assert.Equal(rows, native.MaxDecodeRows);
        Assert.Equal(1, scope.MaxPrefillRows); Assert.Equal(1, native.Begins);
    }

    [Fact]
    public void UpdateAndTrimHaveSeparateCeilingSemantics()
    {
        var native = new Native();
        using var scope = new GgmlHostMoeExpertCacheScope(100, 128, 48, native);
        Assert.Equal(100, scope.MaximumBudgetBytes); Assert.Equal(128, scope.MaxPrefillRows); Assert.Equal(48, scope.LayerCount);
        Assert.Equal(1, scope.MaxDecodeRows); Assert.Equal(1, native.MaxDecodeRows);
        Assert.Equal(11, scope.UpdateBudget(0)); Assert.Equal(0, scope.BudgetBytes);
        Assert.Equal(11, scope.UpdateBudget(100)); Assert.Equal(100, scope.BudgetBytes);
        Assert.Equal(13, scope.Trim(0)); Assert.Equal(100, scope.BudgetBytes);
        Assert.Equal(14, scope.CachedBytes);
        Assert.Equal(new GgmlHostMoeExpertCacheUsage(14, 100, 20, 30, 40), scope.ReadUsage());
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.UpdateBudget(101));
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.UpdateBudget(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => scope.Trim(-1));
        Assert.Equal(2, native.Updates); Assert.Equal(1, native.Trims);
    }

    [Fact]
    public void MaintenanceFailuresKeepOwnerAndLastSuccessfulBudgetForRetry()
    {
        var native = new Native { FailUpdate = true, FailTrim = true, FailEnd = true };
        var scope = new GgmlHostMoeExpertCacheScope(100, 64, 12, native);
        Assert.Throws<InvalidOperationException>(() => scope.UpdateBudget(20));
        Assert.Throws<InvalidOperationException>(() => scope.Trim(0));
        Assert.Throws<InvalidOperationException>(scope.Dispose);
        Assert.True(native.Active); Assert.Equal(100, scope.BudgetBytes);
        native.FailUpdate = native.FailTrim = native.FailEnd = false;
        Assert.Equal(11, scope.UpdateBudget(20)); Assert.Equal(20, scope.BudgetBytes);
        scope.Dispose(); scope.Dispose();
        Assert.False(native.Active); Assert.Equal(2, native.Ends);
        Assert.Throws<ObjectDisposedException>(() => scope.Trim(0));
        Assert.Throws<ObjectDisposedException>(() => scope.UpdateBudget(10));
        Assert.Throws<ObjectDisposedException>(() => scope.ReadUsage());
    }

    [Fact]
    public void CancellationUnwindDisposesConfiguredOwnerOnce()
    {
        var native = new Native();
        Assert.Throws<OperationCanceledException>((Action)(() =>
        {
            using var scope = new GgmlHostMoeExpertCacheScope(0, 1, 1, native);
            throw new OperationCanceledException();
        }));
        Assert.False(native.Active); Assert.Equal(1, native.Ends);
    }
}
