using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpMtpTinyFactAttribute : FactAttribute
{
    public Qwen4ExpMtpTinyFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_FIXTURE")))
            Skip = "Requires the explicitly generated synthetic Qwen4Exp target/head fixture.";
    }
}

public sealed class Qwen4ExpQsaTinyFactAttribute : FactAttribute
{
    public Qwen4ExpQsaTinyFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_QSA") != "1")
            Skip = "Requires the explicit synthetic QSA target and TS_KV_INITIAL_TOKENS=8.";
    }
}

[CollectionDefinition("Qwen4Exp MTP integration", DisableParallelization = true)]
public sealed class Qwen4ExpMtpIntegrationCollection { }

/// <summary>
/// Real managed GGUF loading and selected native backend speculation with synthetic, untrained
/// weights. These are state/ownership and algorithm tests, not model quality,
/// draft acceptance quality or encoder/media quality.
/// </summary>
[Collection("Qwen4Exp MTP integration")]
[Trait("Requires", "Models")]
public sealed class Qwen4ExpMtpIntegrationTests(ITestOutputHelper output)
{
    private static readonly int[] Prompt = [11, 19, 23, 7, 31];

    [Qwen4ExpMtpTinyFact]
    public void DraftHead_F16CacheCatchUpReplayAndTargetIndependence()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        Assert.True(model.HasDraftHead);
        Assert.True(model.SpeculationProfitable);
        var hidden = new float[Prompt.Length * model.SpecFeatureSize];
        var logits = new float[Prompt.Length * model.Config.VocabSize];
        model.SpecForward(Prompt, hidden, logits, allLogitsRows: true);
        var hiddenBefore = (float[])hidden.Clone();
        model.DraftCatchUp(Prompt, hidden, 0);
        var row = hidden[^model.SpecFeatureSize..];
        var draft = Draft(model, 37, row, Prompt.Length);
        AssertFinite(draft.Logits);
        AssertFinite(draft.Hidden);
        Assert.Equal(hiddenBefore, hidden);
        Assert.Equal(Prompt.Length, model.CacheSeqLen);
        float[] continuation = (float[])model.Forward([41]).Clone();
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        Assert.Equal(continuation, model.Forward([41]));

        model.ResetKVCache();
        model.SpecForward(Prompt, hidden, logits, allLogitsRows: true);
        model.DraftCatchUp(Prompt, hidden, 0);
        var replay = Draft(model, 37, hidden[^model.SpecFeatureSize..], Prompt.Length);
        Assert.Equal(draft.Logits, replay.Logits);
        Assert.Equal(draft.Hidden, replay.Hidden);
    }

    [Qwen4ExpMtpTinyFact]
    public void TargetSnapshot_RestoresGdnPleAndRejectsWrongHolderWithoutMutation()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        Assert.True(model.BindSequenceCache("a"));
        model.ForwardRefill(Prompt);
        model.SpecSnapshotRecurrentState();
        model.Forward([43, 47, 53]);
        Assert.True(model.BindSequenceCache("b"));
        model.ForwardRefill([61, 67, 71]);
        int bHead = model.CacheSeqLen;
        Assert.Throws<InvalidOperationException>(model.SpecRestoreRecurrentState);
        Assert.Equal(bHead, model.CacheSeqLen);
        float[] bContinued = (float[])model.Forward([73]).Clone();
        Assert.False(model.BindSequenceCache("a"));
        model.SpecRestoreRecurrentState();
        model.SpecRewindCache(Prompt.Length);
        Assert.Throws<InvalidOperationException>(() => model.SpecRewindCache(Prompt.Length));
        float[] actual = (float[])model.Forward([79, 83]).Clone();

        model.RestorePrimaryCache();
        model.ResetKVCache();
        model.ForwardRefill(Prompt);
        Assert.Equal(actual, model.Forward([79, 83]));
        model.ResetKVCache();
        model.ForwardRefill([61, 67, 71]);
        Assert.Equal(bContinued, model.Forward([73]));

        Assert.False(model.BindSequenceCache("a"));
        model.ResetKVCache();
        Assert.Throws<InvalidOperationException>(model.SpecRestoreRecurrentState);
        Assert.Equal(0, model.CacheSeqLen);
        model.OnSequenceReleased("a");
        Assert.False(model.HasFusedSequenceCache("a"));
        Assert.True(model.HasFusedSequenceCache("b"));
        // Releasing A restores the cold primary at the same four-token B prefix.
        float[] bNextCold = (float[])model.Forward([89]).Clone();
        Assert.False(model.BindSequenceCache("b"));
        Assert.Equal(4, model.CacheSeqLen);
        Assert.Equal(bNextCold, model.Forward([89]));
    }

    [Qwen4ExpMtpTinyFact]
    public void DraftPrivateState_FollowsHolderAndReleasePreservesOtherConversation()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        float[] hA = Hidden(model.SpecFeatureSize, 3), hB = Hidden(model.SpecFeatureSize, 17);
        model.BindSequenceCache("a");
        // Establish the actual native target holder identities before drafting.
        model.Forward([11]);
        var a0 = Draft(model, 11, hA, 0);
        model.BindSequenceCache("b");
        model.Forward([19]);
        var b0 = Draft(model, 19, hB, 0);
        model.BindSequenceCache("a");
        var a1 = Draft(model, 23, a0.Hidden, 1);
        model.BindSequenceCache("b");
        model.OnSequenceReleased("a");
        var b1 = Draft(model, 29, b0.Hidden, 1);
        model.RestorePrimaryCache();
        model.ResetKVCache();
        model.Forward([11]);
        var coldA0 = Draft(model, 11, hA, 0);
        var coldA1 = Draft(model, 23, coldA0.Hidden, 1);
        Assert.Equal(a1.Logits, coldA1.Logits);
        Assert.Equal(a1.Hidden, coldA1.Hidden);
        model.ResetKVCache();
        model.Forward([19]);
        var coldB0 = Draft(model, 19, hB, 0);
        var coldB1 = Draft(model, 29, coldB0.Hidden, 1);
        Assert.Equal(b1.Logits, coldB1.Logits);
        Assert.Equal(b1.Hidden, coldB1.Hidden);
    }

    [Qwen4ExpMtpTinyFact]
    public void DisposingDraftHead_PreservesBorrowedTargetOutputAndRetry()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        float[] before = (float[])model.ForwardRefill(Prompt).Clone();
        Draft(model, 11, Hidden(model.SpecFeatureSize, 3), 0);
        typeof(Qwen4ExpModel).GetMethod("DisposeMtpHead", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(model, null);
        Assert.False(model.HasDraftHead);
        model.ResetKVCache();
        Assert.Equal(before, model.ForwardRefill(Prompt));
        model.Forward([31]);
        AssertFinite(model.Forward([37]));
    }

    [Qwen4ExpMtpTinyFact]
    public void HistoricalMediaPositions_DelayedDraftCatchUpMatchesImmediateChunkSchedule()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        int[][] chunks = [[11, 19], [23, 29], [31, 37]];
        int[][]? axes = [[0,1,2, 0,2,2], [1,3,4, 1,4,4], []];
        var immediate = new List<float[]>();
        var saved = new List<float[]>();
        for (int chunk = 0; chunk < chunks.Length; chunk++)
        {
            if (axes[chunk].Length > 0) model.SetMRoPEPositions((int[])axes[chunk].Clone());
            var h = new float[2 * model.SpecFeatureSize];
            model.SpecForward(chunks[chunk], h, new float[model.Config.VocabSize], false);
            saved.Add(h);
            var logits = new float[model.Config.VocabSize];
            model.DraftCatchUpAndStep(chunks[chunk], h, 2 * chunk, logits, new float[model.SpecFeatureSize]);
            immediate.Add(logits);
        }
        model.ResetKVCache();
        var delayedHidden = new List<float[]>();
        for (int chunk = 0; chunk < chunks.Length; chunk++)
        {
            if (axes[chunk].Length > 0) model.SetMRoPEPositions((int[])axes[chunk].Clone());
            var h = new float[2 * model.SpecFeatureSize];
            model.SpecForward(chunks[chunk], h, new float[model.Config.VocabSize], false);
            Assert.Equal(saved[chunk], h);
            delayedHidden.Add(h);
        }
        for (int chunk = 0; chunk < chunks.Length; chunk++)
        {
            var actual = new float[model.Config.VocabSize];
            model.DraftCatchUpAndStep(chunks[chunk], delayedHidden[chunk], 2 * chunk,
                actual, new float[model.SpecFeatureSize]);
            Assert.Equal(immediate[chunk], actual);
        }
        // This supplies synthetic multi-axis positions directly; it does not
        // load an image/video encoder or establish media task quality.
    }

    [Qwen4ExpMtpTinyFact]
    public void LearnedHead_DraftVerifyRollbackGreedyMatchesPlainWithEngagement()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        const int count = 16;
        var expected = new List<int>();
        float[] logits = model.ForwardRefill(Prompt);
        for (int i = 0; i < count; i++)
        {
            int token = ArgMax(logits);
            expected.Add(token);
            if (i + 1 < count) logits = model.Forward([token]);
        }
        var decoder = new SpeculativeDecoder(model, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead,
            MaxDraftTokens = 3, MaxDraftTokensExplicit = true, MinDraftProb = 0f,
        }) { AdaptiveSpeculation = false, PrefillChunkSize = Prompt.Length };
        // Force the protocol to execute in this untrained fixture; these settings
        // establish correctness and engagement, not a profitability assertion.
        var actual = decoder.GenerateGreedy(Prompt, count, isStopToken: _ => false);
        output.WriteLine($"plain={string.Join(',', expected)} speculative={string.Join(',', actual)} " +
            $"drafted={decoder.TokensDrafted} accepted={decoder.TokensAccepted} verify={decoder.VerifySteps} " +
            $"rollback={decoder.RollbackSteps} plainSteps={decoder.PlainSteps} parked={decoder.ParkedSteps}");
        Assert.True(decoder.TokensDrafted > 0);
        Assert.True(decoder.VerifySteps > 0);
        Assert.True(decoder.RollbackSteps > 0);
        Assert.Equal(expected, actual);
        AssertFinite(model.Forward([97]));
    }

    private static (float[] Logits, float[] Hidden) Draft(Qwen4ExpModel model, int token, float[] hidden, int pos)
    {
        var logits = new float[model.Config.VocabSize];
        var next = new float[model.SpecFeatureSize];
        model.DraftStep(token, hidden, pos, logits, next);
        return (logits, next);
    }

    [Qwen4ExpQsaTinyFact]
    public void QsaRawCache_GrowthPreservesBytesAndRollbackReplaysAllLogits()
    {
        Assert.Equal("8", Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"));
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        int[] middle = [43, 47, 53, 59, 61, 67, 71], end = [73, 79, 83, 89, 97];
        model.BindSequenceCache("growing");
        AllLogits(model, Prompt);
        var a = RawQsa(model);
        Assert.Equal(8, a.Capacity);
        Assert.Contains(a.Bytes, value => value != 0);
        model.SpecEnsureCapacity(9);
        var grown = QsaTensor(model);
        Assert.Equal(16, Field<int>(model, "_kvCacheCapacity"));
        Assert.NotEqual(a.Key, TensorComputePrimitives.GetStoragePointer(grown));
        Assert.Equal(a.Bytes, HostPrefix(grown, a.Bytes.Length));
        float[] mid = AllLogits(model, middle);
        var b = RawQsa(model);
        Assert.Equal(a.Bytes, b.Bytes[..a.Bytes.Length]);
        model.SpecEnsureCapacity(17);
        Assert.Equal(32, Field<int>(model, "_kvCacheCapacity"));
        Assert.NotEqual(b.Key, TensorComputePrimitives.GetStoragePointer(QsaTensor(model)));
        Assert.Equal(b.Bytes, HostPrefix(QsaTensor(model), b.Bytes.Length));
        float[] last = AllLogits(model, end);
        var before = RawQsa(model);
        Assert.Equal(b.Bytes, before.Bytes[..b.Bytes.Length]);
        Assert.Equal(17, Field<int>(model, "_qsaPositionCount"));
        model.SpecSnapshotRecurrentState();
        AllLogits(model, [101, 103, 107]);
        model.SpecRestoreRecurrentState();
        model.SpecRewindCache(17);
        Assert.Equal(before.Bytes, RawQsa(model).Bytes);
        float[] branch = AllLogits(model, [109, 113]);

        cold.BindSequenceCache("cold");
        AllLogits(cold, Prompt);
        cold.SpecEnsureCapacity(9);
        Assert.Equal(mid, AllLogits(cold, middle));
        cold.SpecEnsureCapacity(17);
        Assert.Equal(last, AllLogits(cold, end));
        Assert.Equal(branch, AllLogits(cold, [109, 113]));
        Assert.Equal(RawQsa(cold).Bytes, RawQsa(model).Bytes);
        Assert.Equal(19, Field<int>(model, "_qsaPositionCount"));
        output.WriteLine($"QSA raw cache: capacities 8->16->32, prefix bytes {a.Bytes.Length}->{b.Bytes.Length}->{before.Bytes.Length}; all-logit branch exact.");
    }

    [Qwen4ExpQsaTinyFact]
    public void QsaFirstPromptAndReset_GrowBeforeForwardWithoutExportingInvalidatedState()
    {
        Assert.Equal("8", Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"));
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        int[] first = Enumerable.Range(11, 14).ToArray();
        int[] afterReset = Enumerable.Range(41, 24).ToArray();
        Assert.Equal(8, Field<int>(model, "_kvCacheCapacity"));

        // The first request exceeds the initial allocation before any native
        // QSA entry has been seeded. Compare every logit with a pre-grown model.
        float[] initial = AllLogits(model, first);
        Assert.Equal(16, Field<int>(model, "_kvCacheCapacity"));
        cold.SpecEnsureCapacity(16);
        Assert.Equal(initial, AllLogits(cold, first));
        Assert.Equal(RawQsa(cold).Bytes, RawQsa(model).Bytes);

        model.ResetKVCache();
        Assert.Equal(0, model.CacheSeqLen);
        Assert.False(Field<bool>(model, "_kvCacheHostStale"));
        // Reset invalidates the previous native entry; this larger request
        // forces growth before it can be reseeded, reproducing the video bug.
        float[] reset = AllLogits(model, afterReset);
        Assert.Equal(32, Field<int>(model, "_kvCacheCapacity"));
        cold.ResetKVCache();
        cold.SpecEnsureCapacity(32);
        Assert.Equal(reset, AllLogits(cold, afterReset));
        Assert.Equal(RawQsa(cold).Bytes, RawQsa(model).Bytes);
        Assert.Equal(AllLogits(cold, [79, 83]), AllLogits(model, [79, 83]));
        output.WriteLine("QSA first/reset prompt growth 8->16->32: all logits and raw-key bytes match pre-grown references.");
    }

    [Qwen4ExpQsaTinyFact]
    public void QsaMediaHistory_HolderReleaseAndRewindPreserveOtherRawCache()
    {
        Assert.Equal("8", Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"));
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        int[] text = [11, 19, 23, 29, 31, 37], media = [41, 43, 47, 53, 59, 61];
        int[] axes = [6,2,0, 6,0,1, 6,1,0, 6,0,0, 6,2,1, 6,1,1];
        model.BindSequenceCache("media-a");
        AllLogits(model, text);
        model.SetMRoPEPositions((int[])axes.Clone());
        AllLogits(model, media);
        var a = RawQsa(model);
        var aPositions = Field<int[]>(model, "_qsaPositions")[..36];
        model.SpecSnapshotRecurrentState();
        model.SetMRoPEPositions([7,9,0, 7,9,1, 7,9,2]);
        AllLogits(model, [67, 71, 73]);
        model.BindSequenceCache("other-b");
        AllLogits(model, [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]);
        var b = RawQsa(model);
        Assert.NotEqual(a.Key, b.Key);
        Assert.Throws<InvalidOperationException>(model.SpecRestoreRecurrentState);
        Assert.Equal(b.Bytes, RawQsa(model).Bytes);
        model.BindSequenceCache("media-a");
        model.SpecRestoreRecurrentState();
        model.SpecRewindCache(12);
        Assert.Equal(a.Bytes, RawQsa(model).Bytes);
        Assert.Equal(aPositions, Field<int[]>(model, "_qsaPositions")[..36]);
        model.SetMRoPEPositions([7,0,0, 7,1,0]);
        float[] branch = AllLogits(model, [79, 83]);
        Assert.Equal(new[] {7,0,0, 7,1,0}, Field<int[]>(model, "_qsaPositions")[36..42]);
        float[] textTail = AllLogits(model, [89, 97]);

        cold.BindSequenceCache("reference");
        AllLogits(cold, text);
        cold.SetMRoPEPositions((int[])axes.Clone());
        AllLogits(cold, media);
        cold.SetMRoPEPositions([7,0,0, 7,1,0]);
        Assert.Equal(branch, AllLogits(cold, [79, 83]));
        Assert.Equal(textTail, AllLogits(cold, [89, 97]));
        Assert.Equal(RawQsa(cold).Bytes, RawQsa(model).Bytes);

        model.BindSequenceCache("other-b");
        model.OnSequenceReleased("media-a");
        Assert.False(model.HasFusedSequenceCache("media-a"));
        Assert.True(model.HasFusedSequenceCache("other-b"));
        Assert.Equal(b.Bytes, RawQsa(model).Bytes);
        float[] bNext = AllLogits(model, [163, 167]);
        cold.ResetKVCache();
        AllLogits(cold, [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157]);
        Assert.Equal(bNext, AllLogits(cold, [163, 167]));
        output.WriteLine("QSA ranked THW history, exact live raw-key bytes, wrong-owner refusal, branch rewind, A release/B continuation passed.");
    }

    private static float[] AllLogits(Qwen4ExpModel model, int[] tokens)
    {
        var logits = new float[tokens.Length * model.Config.VocabSize];
        model.SpecForward(tokens, new float[tokens.Length * model.SpecFeatureSize], logits, true);
        AssertFinite(logits);
        return logits;
    }

    private static T Field<T>(Qwen4ExpModel model, string name) => (T)typeof(Qwen4ExpModel)
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static Tensor QsaTensor(Qwen4ExpModel model)
        => Assert.Single(Field<Tensor[]>(model, "_idxKCache").Where(t => t != null));

    private static byte[] HostPrefix(Tensor tensor, int count)
    {
        tensor.Storage.EnsureHostReadable();
        var bytes = new byte[count];
        Marshal.Copy(TensorComputePrimitives.GetStoragePointer(tensor), bytes, 0, bytes.Length);
        return bytes;
    }

    private static unsafe (IntPtr Key, int Capacity, byte[] Bytes) RawQsa(Qwen4ExpModel model)
    {
        Tensor tensor = QsaTensor(model);
        IntPtr key = TensorComputePrimitives.GetStoragePointer(tensor);
        var bytes = new byte[checked((int)tensor.Storage.ByteLength)];
        fixed (byte* destination = bytes)
            Assert.True(GgmlBasicOps.Qwen4ExpCopyQsaCache(key, (IntPtr)destination, bytes.Length, 0));
        int capacity = Field<int>(model, "_kvCacheCapacity");
        int rowBytes = bytes.Length / capacity;
        return (key, capacity, bytes[..checked(rowBytes * model.CacheSeqLen)]);
    }

    private static float[] Hidden(int size, int seed) => Enumerable.Range(0, size)
        .Select(i => (float)Math.Sin((i + seed) * .37)).ToArray();
    private static int ArgMax(float[] row)
    {
        AssertFinite(row);
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }
    private static void AssertFinite(float[] row) => Assert.All(row, value => Assert.True(float.IsFinite(value)));

    private sealed class Fixture : IDisposable
    {
        private readonly string _path;
        private readonly ITestOutputHelper _output;
        internal Fixture(ITestOutputHelper output)
        {
            _output = output;
            _path = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_FIXTURE")!;
            using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(_path, "manifest.json")));
            Assert.True(manifest.RootElement.GetProperty("fixture").GetBoolean());
            foreach (string name in new[] { "target.gguf", "head.gguf" })
            {
                string hash = Hash(Path.Combine(_path, name));
                Assert.Equal(manifest.RootElement.GetProperty("files").GetProperty(name).GetProperty("sha256").GetString(), hash);
                _output.WriteLine($"fixture {name} sha256={hash}");
            }
        }
        internal Qwen4ExpModel Load()
        {
            var backend = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_BACKEND") switch
            {
                null or "" or "GgmlCpu" => BackendType.GgmlCpu,
                "GgmlCuda" => BackendType.GgmlCuda,
                "GgmlMetal" => BackendType.GgmlMetal,
                var unsupported => throw new InvalidOperationException($"Unsupported fixture backend {unsupported}"),
            };
            var model = new Qwen4ExpModel(Path.Combine(_path, "target.gguf"), backend,
                draftGgufPath: Path.Combine(_path, "head.gguf"));
            try
            {
                string native = TestGates.MappedNativeGgmlOpsPath();
                string hash = Hash(native);
                Assert.Equal(Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_NATIVE_SHA256"), hash);
                _output.WriteLine($"native {native} sha256={hash}; backend={backend}");
                Assert.Equal(260, model.Config.VocabSize);
                Assert.Equal(32, model.SpecFeatureSize);
                return model;
            }
            catch { model.Dispose(); throw; }
        }
        private static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        public void Dispose() { }
    }
}
