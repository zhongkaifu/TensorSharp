using System.Diagnostics;
using System.Security.Cryptography;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class DeepSeek41DsparkTinyFactAttribute : FactAttribute
{
    public DeepSeek41DsparkTinyFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_TARGET")))
            Skip = "Requires explicitly pinned tiny DeepSeek V4.1 target and DSpark fixtures.";
    }

    /// <summary>The GGML backend the test constructs; skips unless the process pins it (<see cref="TestGates.GgmlPinSkip"/>).</summary>
    public BackendType GgmlBackend
    {
        get => _ggmlBackend;
        set { _ggmlBackend = value; Skip ??= TestGates.GgmlPinSkip(value); }
    }
    private BackendType _ggmlBackend;
}

[CollectionDefinition("DeepSeek V4.1 DSpark integration", DisableParallelization = true)]
public sealed class DeepSeek41DsparkIntegrationCollection { }

/// <summary>Real native CPU model loading and managed block-speculation lifecycle.
/// Synthetic weights establish execution correctness, not trained draft quality.</summary>
[Collection("DeepSeek V4.1 DSpark integration")]
[Trait("Requires", "Models")]
public sealed class DeepSeek41DsparkIntegrationTests(ITestOutputHelper output)
{
    private static readonly int[] Prompt = [0, 15, 32, 64, 128];

    [DeepSeek41DsparkTinyFact(GgmlBackend = BackendType.GgmlCpu)]
    public void CpuAttachedBlockHead_EngagesAndPreservesSixteenGreedyTokens()
    {
        using var model = Load();
        const int count = 16;
        var expected = new List<int>();
        float[] logits = model.ForwardRefill(Prompt);
        for (int i = 0; i < count; ++i)
        {
            int token = ArgMax(logits);
            expected.Add(token);
            if (i + 1 < count) logits = model.Forward([token]);
        }
        var decoder = new SpeculativeDecoder(model, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.Block,
            MaxDraftTokens = 4, MaxDraftTokensExplicit = true, MinDraftProb = 0f,
        }) { AdaptiveSpeculation = false, PrefillChunkSize = 32 };
        var actual = decoder.GenerateGreedy(Prompt, count, isStopToken: _ => false);
        output.WriteLine($"plain={string.Join(',', expected)} speculative={string.Join(',', actual)} " +
            $"drafted={decoder.TokensDrafted} accepted={decoder.TokensAccepted} verify={decoder.VerifySteps} " +
            $"rollback={decoder.RollbackSteps} plain={decoder.PlainSteps} parked={decoder.ParkedSteps}");
        Assert.True(decoder.TokensDrafted > 0);
        Assert.True(decoder.VerifySteps > 0);
        Assert.Equal(expected, actual);
        // The fixture is untrained. No minimum acceptance/profitability claim.
        int head = model.CacheSeqLen;
        Assert.Throws<InvalidOperationException>(() => model.SpecRewindCache(head + 1));
        Assert.Equal(head, model.CacheSeqLen);
        Assert.Throws<NotSupportedException>(() => model.DraftStep(0, [0], head, new float[256], [0]));
        AssertFinite(model.Forward([7]));
        model.ResetKVCache();
        Assert.Equal(0, model.CacheSeqLen);
        AssertFinite(model.ForwardRefill(Prompt));
    }

    [DeepSeek41DsparkTinyFact(GgmlBackend = BackendType.GgmlCpu)]
    public void VerifyRewindAndTwoSlots_PreserveAcceptedPrefixAndUnrelatedContinuation()
    {
        using var model = Load();
        model.ForwardRefill(Prompt);
        Verify(model, [41, 101, 103]);
        model.SpecRewindCache(Prompt.Length + 1);
        float[] expectedA = (float[])model.Forward([53]).Clone();
        model.ResetKVCache();
        model.ForwardRefill([17, 19, 23]);
        float[] expectedB = (float[])model.Forward([73]).Clone();
        model.ResetKVCache();

        Assert.True(model.BindSequenceCache("a"));
        model.ForwardRefill(Prompt);
        Verify(model, [41, 43, 47]);
        Assert.True(model.BindSequenceCache("b"));
        model.ForwardRefill([17, 19, 23]);
        Assert.False(model.BindSequenceCache("a"));
        int before = model.CacheSeqLen;
        Assert.Throws<InvalidOperationException>(() => model.SpecRewindCache(before + 1));
        Assert.Equal(before, model.CacheSeqLen);
        model.SpecRewindCache(Prompt.Length + 1);
        Compare("accepted_prefix_after_rewind", expectedA, model.Forward([53]));
        model.ResetKVCache();
        model.OnSequenceReleased("a");
        Assert.False(model.HasFusedSequenceCache("a"));
        Assert.False(model.BindSequenceCache("b"));
        Assert.Equal(3, model.CacheSeqLen);
        Compare("other_slot_after_reset_and_release", expectedB, model.Forward([73]));
        model.RestorePrimaryCache();
        Assert.True(model.BindSequenceCache("fresh"));
        Assert.Equal(0, model.CacheSeqLen);
        AssertFinite(model.ForwardRefill(Prompt));
    }

    private static void Verify(DeepSeek4Model model, int[] tokens)
    {
        var all = new float[tokens.Length * model.Config.VocabSize];
        model.SpecForward(tokens, new float[tokens.Length * model.SpecFeatureSize], all, true);
        AssertFinite(all);
    }

    private void Compare(string name, float[] expected, float[] actual)
    {
        AssertFinite(expected); AssertFinite(actual); Assert.Equal(expected.Length, actual.Length);
        double maximum = expected.Zip(actual, (a, b) => Math.Abs((double)a - b)).Max();
        output.WriteLine($"{name}: max_abs={maximum:R}; atol=rtol=2e-5; same prefix/verify/continuation chunk schedule");
        // Existing native inference fixture bound; no broad model or TP claim.
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(Math.Abs((double)actual[i] - expected[i]) <= 2e-5 + 2e-5 * Math.Abs(expected[i]),
                $"{name}[{i}]: {actual[i]:R} versus {expected[i]:R}");
    }

    private static int ArgMax(float[] values)
    {
        AssertFinite(values);
        int best = 0;
        for (int i = 1; i < values.Length; ++i) if (values[i] > values[best]) best = i;
        return best;
    }
    private static void AssertFinite(float[] row) => Assert.All(row, x => Assert.True(float.IsFinite(x)));

    private DeepSeek4Model Load()
    {
        var required = new Dictionary<string, string>
        {
            ["MAX_CONTEXT"] = "1024", ["TS_DSV4_UBATCH"] = "32", ["TS_DSV4_THREADS"] = "2",
            ["TS_DSV41_TP"] = "0", ["TS_DSV41_ENGRAM_THREADS"] = "2", ["TS_DSV41_ENGRAM_WARM"] = "0",
            ["TS_DSV41_RETAINED_CACHE"] = "0", ["TS_DSV41_REWIND_CHECKPOINT"] = "1",
        };
        foreach (var (key, value) in required)
        {
            Assert.Equal(value, Environment.GetEnvironmentVariable(key));
            output.WriteLine($"startup {key}={value}");
        }
        string target = Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_TARGET")!;
        string draft = Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_HEAD")!;
        // Derived from f3917170... by adding only the missing empty BPE merges
        // metadata. All142 original tensor payloads and both sidecars are pinned
        // byte-identical in prepare-dsv41-managed-fixture.py's manifest.
        CheckHash(target, "b455020bd7500c5a835744fb189443451e0849331e15ac72557d58d5eafb13c4");
        CheckHash(Path.Combine(Path.GetDirectoryName(target)!, "deepseek41.config.json"), "159a8b4c221953310a590a40b28c90181d8bc05e421d0f4ec49dda94360e96c0");
        CheckHash(Path.Combine(Path.GetDirectoryName(target)!, "deepseek41.engram.bin"), "d9f9c28124c59c1df587ccd9eef24297c5eefa0aad1bcccc1939b8ded2f5f126");
        CheckHash(draft, "edfdccb348e5e85c714fb8dfe38a61b2324105d1cbd5e14c738324a0940ef600");
        var model = new DeepSeek4Model(target, BackendType.GgmlCpu, draftModelPath: draft);
        try
        {
            string native = TestGates.MappedNativeGgmlOpsPath();
            CheckHash(native, Environment.GetEnvironmentVariable("TS_TEST_DSV41_DSPARK_NATIVE_SHA256")!);
            Assert.True(model.HasDraftHead);
            Assert.Equal(DraftHeadKind.Block, model.DraftHeadKind);
            Assert.Equal(5, model.DraftBlockSize);
            Assert.Equal(256, model.Config.VocabSize);
            Assert.True(model.SpeculationProfitable);
            Assert.True(model.SupportsPerSequenceFusedForward);
            return model;
        }
        catch { model.Dispose(); throw; }
    }

    private void CheckHash(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        string hash = Convert.ToHexStringLower(SHA256.HashData(stream));
        output.WriteLine($"identity {path} sha256={hash}");
        Assert.Equal(expected, hash);
    }
}
