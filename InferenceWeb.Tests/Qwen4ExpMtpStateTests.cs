using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Pure metadata half of speculative rollback. Native GDN/PLE tensor capture
/// and restore need separate native correctness/failure tests; these tests do
/// not call a native snapshot API or simulate its success as numerical proof.
/// </summary>
public class Qwen4ExpMtpStateTests
{
    [Fact]
    public void MetadataSnapshot_CopiesPleHistoryAndRestoresSameOwnerWithoutRewindingEarly()
    {
        using var f = new MetadataFixture();
        object snapshot = f.Capture();
        f.History[0] = 999;
        f.History.Add(444);
        f.Set("_pleNextPos", 25);
        f.Set("_mropeCacheGap", 13);
        f.Position = 23;

        f.Restore(snapshot);

        Assert.Same(f.History, f.Get("_pleHistory"));
        Assert.Equal(new[] { 17, 19, 23 }, f.History);
        Assert.Equal(19, f.Get("_pleNextPos"));
        Assert.Equal(3, f.Get("_mropeCacheGap"));
        // Position changes only in the guarded rewind after native restore.
        Assert.Equal(23, f.Position);
        f.History[1] = -10;
        f.Restore(snapshot);
        Assert.Equal(new[] { 17, 19, 23 }, f.History);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MetadataRestore_RejectsOtherHolderOrResetBeforeChangingItsHistory(bool reset)
    {
        using var f = new MetadataFixture();
        object snapshot = f.Capture();
        if (reset) f.Set("_specResetVersion", 2);
        else f.Set("_gdnConvStateT", new Tensor[2]);
        f.History.Clear(); f.History.Add(314);
        f.Set("_pleNextPos", 80); f.Set("_mropeCacheGap", 9); f.Position = 79;

        Assert.Throws<InvalidOperationException>(() => f.Restore(snapshot));

        Assert.Equal(new[] { 314 }, f.History);
        Assert.Equal(80, f.Get("_pleNextPos"));
        Assert.Equal(9, f.Get("_mropeCacheGap"));
        Assert.Equal(79, f.Position);
    }

    [Fact]
    public void Rewind_RequiresCompletedNativeRestoreAndExactCapturedPositionOnce()
    {
        using var f = new MetadataFixture();
        object snapshot = f.Capture();
        f.Set("_specMetadata", snapshot);
        f.Position = 21;
        Assert.Throws<InvalidOperationException>(() => f.Rewind(17));
        Assert.Equal(21, f.Position);

        // Test the guard after the native path reports completion. This flag is
        // not evidence that native bytes restored; that is a separate fixture.
        f.Set("_specRecurrentRestored", true);
        Assert.Throws<InvalidOperationException>(() => f.Rewind(19));
        Assert.Equal(21, f.Position);
        f.Rewind(17);
        Assert.Equal(17, f.Position);
        Assert.Throws<InvalidOperationException>(() => f.Rewind(17));
    }

    [Fact]
    public void Rewind_RejectsChangedHolderEvenAtSamePosition()
    {
        using var f = new MetadataFixture();
        f.Set("_specMetadata", f.Capture());
        f.Set("_specRecurrentRestored", true);
        f.Set("_gdnConvStateT", new Tensor[2]);
        Assert.Throws<InvalidOperationException>(() => f.Rewind(17));
        Assert.Equal(17, f.Position);
    }

    private sealed class MetadataFixture : IDisposable
    {
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        internal readonly List<int> History = new() { 17, 19, 23 };
        internal MetadataFixture()
        {
            // This metadata-only fixture has no QSA layers. The ordinary
            // constructor supplies Config before any rewind capability query.
            Qwen4ExpMtpMathTests.Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig { NumLayers = 0 });
            Set("_gdnConvStateT", new Tensor[2]);
            Set("_pleHistory", History);
            Set("_pleNextPos", 19);
            Set("_mropeCacheGap", 3);
            Set("_specResetVersion", 1);
            Position = 17;
        }
        internal int Position
        {
            get => Model.CacheSeqLen;
            set => Qwen4ExpMtpMathTests.Set(typeof(ModelBase), Model, "_cacheSeqLen", value);
        }
        internal void Set(string field, object value) => Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), Model, field, value);
        internal object? Get(string field) => typeof(Qwen4ExpModel).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Model);
        internal object Capture() => Qwen4ExpMtpMathTests.Invoke(Model, "CaptureSpecMetadata", Array.Empty<object>())!;
        internal void Restore(object metadata) => Qwen4ExpMtpMathTests.Invoke(Model, "RestoreSpecMetadata", new[] { metadata });
        internal void Rewind(int position) => Qwen4ExpMtpMathTests.Invoke(Model, "SpecRewindCache", new object[] { position });
        public void Dispose() => GC.SuppressFinalize(Model);
    }
}
