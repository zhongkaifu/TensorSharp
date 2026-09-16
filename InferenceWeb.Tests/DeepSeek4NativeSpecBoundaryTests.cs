using System.Diagnostics;
using System.Security.Cryptography;
using TensorSharp.GGML;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Managed array boundaries around the real native DSpark executor.
/// Empty-output rows are safe to run against the old binding as red controls;
/// nonempty short buffers must only run after the checked binding is built.</summary>
[Collection("DeepSeek V4.1 DSpark integration")]
public sealed class DeepSeek4NativeSpecBoundaryTests(ITestOutputHelper output)
{
    [Fact]
    public void NullArraysFailBeforeNativeEntry()
    {
        Assert.Throws<ArgumentNullException>(() => GgmlDeepSeek4Native.ForwardSpec(IntPtr.Zero, null!, []));
        Assert.Throws<ArgumentNullException>(() => GgmlDeepSeek4Native.ForwardSpec(IntPtr.Zero, [1], null!));
        Assert.Throws<ArgumentNullException>(() => GgmlDeepSeek4Native.DsparkDraft(IntPtr.Zero, 1, null!, []));
        Assert.Throws<ArgumentNullException>(() => GgmlDeepSeek4Native.DsparkDraft(IntPtr.Zero, 1, [], null!));
    }

    [Fact]
    public void NullHandlePreservesNormalUnavailableResult()
    {
        Assert.False(GgmlDeepSeek4Native.ForwardSpec(IntPtr.Zero, [1], []));
        Assert.Equal(0, GgmlDeepSeek4Native.DsparkDraft(IntPtr.Zero, 1, [], []));
    }

    [DeepSeek41DsparkTinyFact, Trait("Requires", "Models")]
    public void EmptyForwardOutputIsRejected()
    {
        using var model = Load();
        Assert.Equal("logitsOut", Assert.Throws<ArgumentException>(() =>
            GgmlDeepSeek4Native.ForwardSpec(model.Handle, [7], [])).ParamName);
        Assert.Equal(0, GgmlDeepSeek4Native.NPast(model.Handle));
    }

    [DeepSeek41DsparkTinyFact, Trait("Requires", "Models")]
    public void EmptyDraftTokenOutputIsRejected()
    {
        using var model = Load();
        Seed(model.Handle);
        var confidence = Enumerable.Repeat(-123f, 5).ToArray();
        Assert.Equal("toksOut", Assert.Throws<ArgumentException>(() =>
            GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, [], confidence)).ParamName);
        Assert.All(confidence, x => Assert.Equal(-123f, x));
        Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
    }

    [DeepSeek41DsparkTinyFact, Trait("Requires", "Models")]
    public void EmptyDraftConfidenceOutputIsRejected()
    {
        using var model = Load();
        Seed(model.Handle);
        var tokens = Enumerable.Repeat(-123, 5).ToArray();
        Assert.Equal("confOut", Assert.Throws<ArgumentException>(() =>
            GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, tokens, [])).ParamName);
        Assert.All(tokens, x => Assert.Equal(-123, x));
        Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
    }

    [DeepSeek41DsparkTinyFact, Trait("Requires", "Models")]
    public void ShortVerifyBuffersLeaveStateUntouchedAndFullRowsRetryMatchesCold()
    {
        using var model = Load();
        Seed(model.Handle);
        int vocabulary = GgmlDeepSeek4Native.VocabSize(model.Handle);
        int[] tokens = [41, 43, 47];
        foreach (int length in new[] { vocabulary, tokens.Length * vocabulary - 1 })
        {
            var shortOutput = Enumerable.Repeat(-123f, length).ToArray();
            Assert.Throws<ArgumentException>(() => GgmlDeepSeek4Native.ForwardSpec(model.Handle, tokens, shortOutput));
            Assert.All(shortOutput, x => Assert.Equal(-123f, x));
            Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
            Assert.Equal(new[] { 41, 43, 47 }, tokens);
        }
        int required = tokens.Length * vocabulary;
        var actual = Enumerable.Repeat(-123f, required + 7).ToArray();
        Assert.True(GgmlDeepSeek4Native.ForwardSpec(model.Handle, tokens, actual));
        Assert.Equal(8, GgmlDeepSeek4Native.NPast(model.Handle));
        Assert.All(actual.Take(required), x => Assert.True(float.IsFinite(x)));
        Assert.All(actual.Skip(required), x => Assert.Equal(-123f, x));
        Assert.True(GgmlDeepSeek4Native.ResetChecked(model.Handle));
        Seed(model.Handle);
        var expected = new float[required];
        Assert.True(GgmlDeepSeek4Native.ForwardSpec(model.Handle, tokens, expected));
        Assert.Equal(expected, actual.Take(required));
    }

    [DeepSeek41DsparkTinyFact, Trait("Requires", "Models")]
    public void ShortDraftBuffersLeaveBothOutputsUntouchedAndRetryPreservesHead()
    {
        using var model = Load();
        Seed(model.Handle);
        int block = GgmlDeepSeek4Native.DsparkBlockSize(model.Handle);
        var fullTokens = Enumerable.Repeat(-123, block + 7).ToArray();
        var fullConfidence = Enumerable.Repeat(-123f, block + 7).ToArray();
        var shortTokens = Enumerable.Repeat(-123, block - 1).ToArray();
        var shortConfidence = Enumerable.Repeat(-123f, block - 1).ToArray();
        Assert.Throws<ArgumentException>(() => GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, shortTokens, fullConfidence));
        Assert.Throws<ArgumentException>(() => GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, fullTokens, shortConfidence));
        Assert.All(fullTokens.Concat(shortTokens), x => Assert.Equal(-123, x));
        Assert.All(fullConfidence.Concat(shortConfidence), x => Assert.Equal(-123f, x));
        Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
        Assert.Equal(block, GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, fullTokens, fullConfidence));
        Assert.All(fullTokens.Take(block), x => Assert.InRange(x, 0, 255));
        Assert.All(fullConfidence.Take(block), x => Assert.True(float.IsFinite(x) && x >= 0 && x <= 1));
        Assert.All(fullTokens.Skip(block), x => Assert.Equal(-123, x));
        Assert.All(fullConfidence.Skip(block), x => Assert.Equal(-123f, x));
        Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
        var retryTokens = new int[block];
        var retryConfidence = new float[block];
        Assert.Equal(block, GgmlDeepSeek4Native.DsparkDraft(model.Handle, 17, retryTokens, retryConfidence));
        Assert.Equal(fullTokens.Take(block), retryTokens);
        Assert.Equal(fullConfidence.Take(block), retryConfidence);
        Assert.Equal(5, GgmlDeepSeek4Native.NPast(model.Handle));
    }

    private static void Seed(IntPtr handle)
        => Assert.True(GgmlDeepSeek4Native.Forward(handle, [0, 15, 32, 64, 128], new float[256]));

    private NativeModel Load()
    {
        foreach (var (name, value) in new Dictionary<string, string>
        {
            ["TS_DSV41_TP"] = "0", ["TS_DSV41_ENGRAM_THREADS"] = "2",
            ["TS_DSV41_ENGRAM_WARM"] = "0", ["TS_DSV41_RETAINED_CACHE"] = "0",
            ["TS_DSV41_REWIND_CHECKPOINT"] = "1",
        }) Assert.Equal(value, Environment.GetEnvironmentVariable(name));
        string target = Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_TARGET")!;
        string head = Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_HEAD")!;
        CheckHash(target, "b455020bd7500c5a835744fb189443451e0849331e15ac72557d58d5eafb13c4");
        CheckHash(head, "edfdccb348e5e85c714fb8dfe38a61b2324105d1cbd5e14c738324a0940ef600");
        string directory = Path.GetDirectoryName(target)!;
        CheckHash(Path.Combine(directory, "deepseek41.config.json"), "159a8b4c221953310a590a40b28c90181d8bc05e421d0f4ec49dda94360e96c0");
        CheckHash(Path.Combine(directory, "deepseek41.engram.bin"), "d9f9c28124c59c1df587ccd9eef24297c5eefa0aad1bcccc1939b8ded2f5f126");
        var handle = GgmlDeepSeek4Native.LoadModelWithDspark(target, 1, 1024, 32, 2, head, backendName: "CPU");
        Assert.NotEqual(IntPtr.Zero, handle);
        try
        {
            string native = TestGates.MappedNativeGgmlOpsPath();
            CheckHash(native, Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_NATIVE_SHA256")!);
            Assert.Equal(256, GgmlDeepSeek4Native.VocabSize(handle));
            Assert.Equal(5, GgmlDeepSeek4Native.DsparkBlockSize(handle));
            return new NativeModel(handle);
        }
        catch { GgmlDeepSeek4Native.Free(handle); throw; }
    }

    private void CheckHash(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        string hash = Convert.ToHexStringLower(SHA256.HashData(stream));
        output.WriteLine($"identity {path} sha256={hash}");
        Assert.Equal(expected, hash);
    }

    private sealed class NativeModel(IntPtr handle) : IDisposable
    {
        public IntPtr Handle { get; } = handle;
        public void Dispose() => GgmlDeepSeek4Native.Free(Handle);
    }
}
