using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Complete QSA coordinate metadata and holder ownership. Native raw
/// indexer-key bytes and actual sparse attention require separate numerical tests.</summary>
public sealed class Qwen4ExpQsaHistoryTests
{
    [Fact]
    public void TextMediaAndDelayedTextPreserveCompleteHistoryAndCallerCoordinates()
    {
        int[] history = Enumerable.Repeat(-123, 30).ToArray();
        Qwen4ExpModel.WriteQsaPositions(history, 0, 2, 0, null!);
        int[] media = [2,10,20, 2,11,20, 3,10,21];
        int[] original = (int[])media.Clone();
        Qwen4ExpModel.WriteQsaPositions(history, 2, 3, 2, media);
        Qwen4ExpModel.WriteQsaPositions(history, 5, 2, 4, null!);
        Assert.Equal(new[] {0,0,0, 1,1,1, 2,10,20, 2,11,20, 3,10,21, 4,4,4, 5,5,5}, history.Take(21));
        Assert.All(history.Skip(21), x => Assert.Equal(-123, x));
        Assert.Equal(original, media);
        media[0] = 999;
        Assert.Equal(2, history[6]);
    }

    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)]
    [InlineData(4)] [InlineData(5)] [InlineData(6)] [InlineData(7)]
    public void InvalidWriteNeverModifiesAnyHistoryCell(int kind)
    {
        int[] history = Enumerable.Range(0, 12).ToArray(), before = (int[])history.Clone();
        int start = 1, count = 2, rope = 10;
        int[]? media = null;
        switch (kind)
        {
            case 0: start = -1; break;
            case 1: count = 0; break;
            case 2: start = 3; break;
            case 3: rope = -1; break;
            case 4: rope = int.MaxValue; break; // invalid second scalar row
            case 5: media = [1,2,3]; break;
            case 6: media = [1,2,3, 4,5,-1]; break; // invalid final axis
            case 7: start = int.MaxValue; count = int.MaxValue; break;
        }
        Assert.Throws<ArgumentException>(() => Qwen4ExpModel.WriteQsaPositions(history, start, count, rope, media!));
        Assert.Equal(before, history);
    }

    [Fact]
    public void CausalCoordinateOrderIsTimeThenHeightThenWidth()
    {
        int[] positions = [2,100,100, 3,0,0, 3,0,1, 3,1,0, 3,1,0];
        Assert.True(Qwen4ExpModel.CompareQsaPositions(positions, 0, 1) < 0);
        Assert.True(Qwen4ExpModel.CompareQsaPositions(positions, 1, 2) < 0);
        Assert.True(Qwen4ExpModel.CompareQsaPositions(positions, 2, 3) < 0);
        Assert.Equal(0, Qwen4ExpModel.CompareQsaPositions(positions, 3, 4));
        Assert.True(Qwen4ExpModel.CompareQsaPositions(positions, 3, 2) > 0);
    }

    [Fact]
    public void HolderSwitchRestoresItsOwnCompleteHistoryAndSpecRewindReplacesOnlyTail()
    {
        using var f = new Fixture();
        var a = new int[36];
        Qwen4ExpModel.WriteQsaPositions(a, 0, 9, 0, null!);
        f.Set("_qsaPositions", a); f.Set("_qsaPositionCount", 5); f.Position = 5;
        object metadata = f.Call("CaptureSpecMetadata");
        object holderA = f.Call("SnapshotActiveCache");
        int[] b = new int[24];
        Qwen4ExpModel.WriteQsaPositions(b, 0, 3, 100, null!);
        f.Set("_qsaPositions", b); f.Set("_qsaPositionCount", 3); f.Position = 3;
        f.Set("_gdnConvStateT", new Tensor[1]);
        object holderB = f.Call("SnapshotActiveCache");
        f.Call("LoadCacheHolder", holderA);
        Assert.Same(a, f.Get("_qsaPositions")); Assert.Equal(5, f.Get("_qsaPositionCount"));
        f.Position = 9; f.Set("_qsaPositionCount", 9);
        f.Set("_specMetadata", metadata); f.Set("_specRecurrentRestored", true);
        // This flag tests the post-native-restore metadata contract only.
        f.Model.SpecRewindCache(5);
        Assert.Equal(5, f.Position); Assert.Equal(5, f.Get("_qsaPositionCount"));
        Qwen4ExpModel.WriteQsaPositions(a, 5, 2, 50, null!);
        f.Set("_qsaPositionCount", 7); f.Position = 7;
        Assert.Equal(Enumerable.Range(0,5).SelectMany(x => new[]{x,x,x}), a.Take(15));
        Assert.Equal(new[] {50,50,50,51,51,51}, a.Skip(15).Take(6));
        f.Call("LoadCacheHolder", holderB);
        Assert.Same(b, f.Get("_qsaPositions")); Assert.Equal(3, f.Get("_qsaPositionCount"));
        Assert.Equal(3, f.Position);
        Assert.Equal(new[] {100,100,100,101,101,101,102,102,102}, b.Take(9));
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void OtherOwnerOrResetCannotRewindItsHistory(bool reset)
    {
        using var f = new Fixture();
        int[] history = Enumerable.Range(0, 18).ToArray();
        f.Set("_qsaPositions", history); f.Set("_qsaPositionCount", 3); f.Position = 3;
        f.Set("_specMetadata", f.Call("CaptureSpecMetadata"));
        f.Set("_specRecurrentRestored", true);
        if (reset) f.Set("_specResetVersion", 2);
        else f.Set("_gdnConvStateT", new Tensor[1]);
        f.Position = 5; f.Set("_qsaPositionCount", 5);
        Assert.Throws<InvalidOperationException>(() => f.Model.SpecRewindCache(3));
        Assert.Equal(5, f.Position); Assert.Equal(5, f.Get("_qsaPositionCount"));
        Assert.Equal(Enumerable.Range(0,18), history);
    }

    private sealed class Fixture : IDisposable
    {
        internal Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        internal Fixture()
        {
            Qwen4ExpMtpMathTests.Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig { NumLayers = 1 });
            Set("_isRecurrent", new[] {false}); Set("_indexerHeads", 2); Set("_indexerHeadDim", 8);
            Set("_compressRatios", new[] {4}); Set("_gdnConvStateT", new Tensor[1]); Set("_specResetVersion", 1);
        }
        internal int Position { get => Model.CacheSeqLen; set => Qwen4ExpMtpMathTests.Set(typeof(ModelBase), Model, "_cacheSeqLen", value); }
        internal void Set(string name, object value) => Qwen4ExpMtpMathTests.Set(typeof(Qwen4ExpModel), Model, name, value);
        internal object? Get(string name) => typeof(Qwen4ExpModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(Model);
        internal object Call(string name, params object[] args) => Qwen4ExpMtpMathTests.Invoke(Model, name, args)!;
        public void Dispose() => GC.SuppressFinalize(Model);
    }
}
