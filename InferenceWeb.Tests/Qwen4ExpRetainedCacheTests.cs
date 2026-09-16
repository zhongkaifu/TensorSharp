// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen4ExpRetainedSplitFactAttribute : FactAttribute
{
    public Qwen4ExpRetainedSplitFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_LAYER_SPLIT") != "2")
            Skip = "Requires an explicitly reserved pair of CUDA GPUs and TS_TEST_QWEN4EXP_LAYER_SPLIT=2.";
    }
}

/// <summary>
/// Retained-prefix reuse for qwen4exp on the synthetic target + shared-MTP
/// fixture: a released conversation's holder re-keyed for its follow-up turn, a
/// shared-prefix checkpoint cloned for new chats, the draft head's private state
/// travelling with both, the QSA position history and raw indexer cache
/// travelling with both, the retention budget on real holders, and the engine
/// taking those paths. These are exactness and ownership tests on untrained
/// weights, not model-quality tests.
/// </summary>
[Collection("Qwen4Exp MTP integration")]
[Trait("Requires", "Models")]
public sealed class Qwen4ExpRetainedCacheTests(ITestOutputHelper output)
{
    private static readonly int[] Prompt = [11, 19, 23, 7, 31, 37, 41, 43];
    private static readonly int[] Media = [47, 53, 59, 61, 67, 71];
    private static readonly int[] MediaAxes = [6,2,0, 6,0,1, 6,1,0, 6,0,0, 6,2,1, 6,1,1];
    private static readonly int[] AfterMedia = [73, 79, 83, 89];
    private static readonly int[] AfterMediaAxes = [7,9,0, 7,9,1, 7,9,2, 7,9,3];
    private static readonly int[] Other = [101, 103, 107, 109, 113, 127, 131, 137];
    private static readonly int[] Suffix = [97, 149, 151];

    [Qwen4ExpRetainedSplitFact]
    public unsafe void LayerSplitCheckpoint_PreservesAllDeviceStateAndFullLogits()
    {
        Assert.Equal("GgmlCuda", Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_BACKEND"));
        Assert.True(GgmlBasicOps.GetGpuDeviceCount(GgmlBackendType.Cuda) >= 2);
        string previousSplit = Environment.GetEnvironmentVariable("TS_Q4E_LAYER_SPLIT");
        Environment.SetEnvironmentVariable("TS_Q4E_LAYER_SPLIT", "1,1");
        try
        {
            using var fixture = new Fixture(output);
            using var model = fixture.Load(layerSplitDegree: 2);
            using var cold = fixture.Load(layerSplitDegree: 2);
            Assert.Equal(0, model.DeviceForLayer(0));
            Assert.Equal(1, model.DeviceForLayer(1));
            Assert.Equal(1, Field<Qwen4ExpMtpConfig>(model, "_mtpConfig").Device);
            Assert.True(model.SupportsPrefixCheckpoints);
            Assert.True(model.BindSequenceCache("a"));
            RunConversation(model);
            Draft(model, 139, Hidden(model.SpecFeatureSize, 5), 0);
            AssertSplitStatePlacement(model);
            byte[] sourceKeys = RawQsa(model).Bytes;
            Assert.True(model.TryCheckpointActiveCache("checkpoint"));
            Assert.True(model.IsRetainedCheckpoint("checkpoint"));
            Assert.False(model.TryRebindRetainedCache("checkpoint", "moved"));
            Assert.True(model.TryCloneRetainedCache("checkpoint", "a2"));
            var expectedDraft = Draft(model, 149, Hidden(model.SpecFeatureSize, 7), 1);
            float[][] originalContinuation = Continue(model, Suffix, 3);
            Assert.True(model.BindSequenceCache("b"));
            model.ForwardRefill(Other);
            Draft(model, 151, Hidden(model.SpecFeatureSize, 11), 0);
            float[] bBefore = (float[])model.Forward([157]).Clone();

            Assert.True(model.TryCloneRetainedCache("checkpoint", "a3"));
            Assert.False(model.BindSequenceCache("a2"));
            Assert.Equal(sourceKeys, RawQsaAfterSeed(model));
            var actualDraft = Draft(model, 149, Hidden(model.SpecFeatureSize, 7), 1);
            Assert.Equal(expectedDraft.Logits, actualDraft.Logits);
            Assert.Equal(expectedDraft.Hidden, actualDraft.Hidden);
            float[][] cloneContinuation = Continue(model, Suffix, 3);
            AssertSplitStatePlacement(model);
            Assert.False(model.BindSequenceCache("a3"));
            float[][] anotherCloneContinuation = Continue(model, Suffix, 3);
            RunConversation(cold);
            float[][] coldContinuation = Continue(cold, Suffix, 3);
            for (int step = 0; step < coldContinuation.Length; ++step)
            {
                Assert.Equal(coldContinuation[step], originalContinuation[step]);
                Assert.Equal(coldContinuation[step], cloneContinuation[step]);
                Assert.Equal(coldContinuation[step], anotherCloneContinuation[step]);
            }
            Assert.False(model.BindSequenceCache("b"));
            float[] bAfter = (float[])model.Forward([163]).Clone();
            cold.ResetKVCache();
            cold.ForwardRefill(Other);
            Assert.Equal(bBefore, cold.Forward([157]));
            Assert.Equal(bAfter, cold.Forward([163]));
            foreach (string id in new[] { "a", "b", "a2", "a3" }) model.OnSequenceReleased(id);
            model.DiscardRetainedCache("checkpoint");
            Assert.Equal(0, model.RetainedCacheCount);
            model.RestorePrimaryCache();
            output.WriteLine("Physical two-device public checkpoint/clone: GDN/PLE rank0, QSA/MTP rank1; A/B/A independent head and full-logit cold parity.");
        }
        finally { Environment.SetEnvironmentVariable("TS_Q4E_LAYER_SPLIT", previousSplit); }
    }

    private static byte[] RawQsaAfterSeed(Qwen4ExpModel model)
    {
        // A newly copied holder has host-authoritative raw keys until its first
        // forward creates a native entry; inspect exactly its written host prefix.
        Tensor tensor = QsaTensor(model);
        tensor.Storage.EnsureHostReadable();
        int bytes = checked((int)(tensor.Storage.ByteLength / Field<int>(model, "_kvCacheCapacity") * model.CacheSeqLen));
        var result = new byte[bytes];
        Marshal.Copy(TensorComputePrimitives.GetStoragePointer(tensor), result, 0, bytes);
        return result;
    }

    private static unsafe void AssertSplitStatePlacement(Qwen4ExpModel model)
    {
        Tensor conv = Field<Tensor[]>(model, "_gdnConvStateT")[0];
        Tensor ssm = Field<Tensor[]>(model, "_gdnStateT")[0];
        Check(TensorComputePrimitives.GetStoragePointer(conv), ((conv.Storage.ByteLength + 255) & ~255L) + ssm.Storage.ByteLength, 0);
        Tensor qsa = QsaTensor(model);
        Check(TensorComputePrimitives.GetStoragePointer(qsa), qsa.Storage.ByteLength, 1);
        float[] ple = Field<float[]>(model, "_pleConvState");
        Assert.NotEmpty(ple);
        Check(Marshal.UnsafeAddrOfPinnedArrayElement(ple, 0), (long)ple.Length * sizeof(float), 0);

        static void Check(IntPtr key, long bytes, int rank)
        {
            var copy = new byte[checked((int)bytes)];
            fixed (byte* pointer = copy)
            {
                Assert.False(GgmlBasicOps.Qwen4ExpCopyQsaCache(key, (IntPtr)pointer, bytes, 1 - rank));
                Assert.True(GgmlBasicOps.Qwen4ExpCopyQsaCache(key, (IntPtr)pointer, bytes, rank));
            }
        }
    }

    [Qwen4ExpMtpTinyFact]
    public void Checkpoint_MissingAuthoritativeRecurrentEntryDoesNotPublishStaleHostState()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        model.BindSequenceCache("source");
        model.ForwardRefill(Prompt);
        Tensor recurrent = Assert.Single(Field<Tensor[]>(model, "_gdnConvStateT").Where(t => t != null));
        GgmlBasicOps.Qwen4ExpInvalidateSeqState(TensorComputePrimitives.GetStoragePointer(recurrent));
        var error = Assert.Throws<InvalidOperationException>(() => model.TryCheckpointActiveCache("invalid"));
        Assert.Contains("authoritative GDN state", error.Message);
        Assert.Equal(0, model.RetainedCacheCount);
        Assert.False(model.TryCloneRetainedCache("invalid", "clone"));
        Assert.True(model.HasFusedSequenceCache("source"));
    }

    [Qwen4ExpMtpTinyFact]
    public void ReleasedConversation_RebindsAfterAnotherRequest_BitExact_WithQsaHistory()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        Assert.True(model.SupportsRetainedFusedCache);
        Assert.True(model.SupportsExactFusedCacheReuse);
        Assert.False(model.SupportsKVCacheTruncation);

        // Conversation A: text, a media chunk with (T,H,W) positions, text with
        // explicit positions - so the QSA history, the rotary gap and the PLE window
        // are all non-trivial by the time it is released.
        Assert.True(model.BindSequenceCache("a"));
        RunConversation(model);
        int retainedLength = model.CacheSeqLen;
        Assert.Equal(Prompt.Length + Media.Length + AfterMedia.Length, retainedLength);
        int[] positionsAtRelease = Field<int[]>(model, "_qsaPositions")[..(3 * retainedLength)];
        byte[] rawQsaAtRelease = RawQsa(model).Bytes;
        int gapAtRelease = Field<int>(model, "_mropeCacheGap");

        Assert.True(model.RetainSequenceCache("a"));
        Assert.False(model.HasFusedSequenceCache("a"));
        Assert.Equal(1, model.RetainedCacheCount);
        Assert.True(model.RetainedCacheBytes("a") > 0);
        // Exact prefix only: the retained tokens, no more and no fewer.
        Assert.True(model.CanReuseRetainedPrefix("a", retainedLength, retainedLength));
        Assert.False(model.CanReuseRetainedPrefix("a", retainedLength, retainedLength - 1));
        Assert.False(model.CanReuseRetainedPrefix("a", retainedLength + 1, retainedLength + 1));
        Assert.False(model.CanReuseRetainedPrefix("missing", retainedLength, retainedLength));

        // B runs in between, in its own holder, and is not disturbed by A's return.
        Assert.True(model.BindSequenceCache("b"));
        model.ForwardRefill(Other);
        float[] bFirst = (float[])model.Forward([139]).Clone();

        Assert.True(model.TryRebindRetainedCache("a", "a2"));
        Assert.False(model.TryRebindRetainedCache("a", "a3"));
        Assert.False(model.CanReuseRetainedPrefix("a", retainedLength, retainedLength));
        Assert.True(model.HasFusedSequenceCache("a2"));
        Assert.Equal(0, model.RetainedCacheCount);
        Assert.False(model.BindSequenceCache("a2"));
        Assert.Equal(retainedLength, model.CacheSeqLen);
        Assert.Equal(retainedLength, Field<int>(model, "_qsaPositionCount"));
        Assert.Equal(positionsAtRelease, Field<int[]>(model, "_qsaPositions")[..(3 * retainedLength)]);
        Assert.Equal(rawQsaAtRelease, RawQsa(model).Bytes);
        Assert.Equal(gapAtRelease, Field<int>(model, "_mropeCacheGap"));

        float[][] actual = Continue(model, Suffix, 3);
        int finalLength = retainedLength + Suffix.Length + 3;
        Assert.Equal(finalLength, model.CacheSeqLen);
        byte[] rawQsaAfter = RawQsa(model).Bytes;
        int[] positionsAfter = Field<int[]>(model, "_qsaPositions")[..(3 * finalLength)];

        Assert.False(model.BindSequenceCache("b"));
        float[] bNext = (float[])model.Forward([149]).Clone();

        // The uninterrupted continuation, on a model that never released anything.
        Assert.True(cold.BindSequenceCache("reference"));
        RunConversation(cold);
        float[][] expected = Continue(cold, Suffix, 3);
        for (int step = 0; step < expected.Length; ++step)
            Assert.Equal(expected[step], actual[step]);
        Assert.Equal(RawQsa(cold).Bytes, rawQsaAfter);
        Assert.Equal(finalLength, cold.CacheSeqLen);
        Assert.Equal(Field<int[]>(cold, "_qsaPositions")[..(3 * finalLength)], positionsAfter);

        cold.ResetKVCache();
        cold.ForwardRefill(Other);
        Assert.Equal(bFirst, cold.Forward([139]));
        Assert.Equal(bNext, cold.Forward([149]));

        // A finished turn of the rebound conversation is retained again under its
        // new key; releasing B does not touch it.
        Assert.True(model.RetainSequenceCache("a2"));
        model.OnSequenceReleased("a2");
        model.OnSequenceReleased("b");
        Assert.True(model.CanReuseRetainedPrefix("a2", finalLength, finalLength));
        model.DiscardRetainedCache("a2");
        Assert.Equal(0, model.RetainedCacheCount);
        Assert.False(model.CanReuseRetainedPrefix("a2", finalLength, finalLength));
        model.RestorePrimaryCache();
        output.WriteLine($"A->B->A rebind: retained {retainedLength} tokens, 3 continuation steps bit-exact, QSA history {positionsAtRelease.Length / 3} cells carried.");
    }

    [Qwen4ExpMtpTinyFact]
    public void SharedPrefixCheckpoint_ClonesEqualColdPrefill_AndTheSourceIsUntouched()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        Assert.True(model.SupportsPrefixCheckpoints);
        int[] shared = Prompt.Concat(Other).ToArray();
        int[] first = [151, 157, 163];
        int[] second = [167, 173, 179];

        // The first chat reaches the end of the shared prefix (its holder is
        // device-dirty: the fused span wrote its KV, GDN, PLE and QSA state on the
        // device), the draft head has private state for it, and it is checkpointed
        // there.
        Assert.True(model.BindSequenceCache("chat-1"));
        var hidden = new float[shared.Length * model.SpecFeatureSize];
        model.SpecForward(shared, hidden, new float[shared.Length * model.Config.VocabSize], allLogitsRows: true);
        model.DraftCatchUp(shared, hidden, 0);
        var sourceDraft = Draft(model, first[0], hidden[^model.SpecFeatureSize..], shared.Length);
        int[] positionsAtBoundary = Field<int[]>(model, "_qsaPositions")[..(3 * shared.Length)];
        Assert.True(model.TryCheckpointActiveCache("ckpt"));
        Assert.False(model.TryCheckpointActiveCache("ckpt"));
        Assert.Equal(1, model.RetainedCacheCount);
        Assert.True(model.IsRetainedCheckpoint("ckpt"));
        Assert.False(model.TryRebindRetainedCache("ckpt", "moved")); // cloned, never moved

        // Chat 1 carries on past the boundary: the checkpoint must own its bytes.
        float[][] firstContinuation = Continue(model, first, 2);

        // Chat 2 starts from a clone: QSA history, draft state and all.
        Assert.True(model.TryCloneRetainedCache("ckpt", "chat-2"));
        Assert.True(model.HasFusedSequenceCache("chat-2"));
        Assert.False(model.BindSequenceCache("chat-2"));
        Assert.Equal(shared.Length, model.CacheSeqLen);
        Assert.Equal(shared.Length, Field<int>(model, "_qsaPositionCount"));
        Assert.Equal(positionsAtBoundary, Field<int[]>(model, "_qsaPositions")[..(3 * shared.Length)]);
        var cloneDraft = Draft(model, first[0], hidden[^model.SpecFeatureSize..], shared.Length);
        Assert.Equal(sourceDraft.Logits, cloneDraft.Logits);
        Assert.Equal(sourceDraft.Hidden, cloneDraft.Hidden);
        float[][] secondFromClone = Continue(model, second, 3);
        byte[] rawQsaChat2 = RawQsa(model).Bytes;

        // And a third chat from the SAME checkpoint after chat 2 decoded on its copy.
        Assert.True(model.TryCloneRetainedCache("ckpt", "chat-3"));
        Assert.False(model.BindSequenceCache("chat-3"));
        float[][] secondFromCloneAgain = Continue(model, second, 3);
        for (int step = 0; step < secondFromClone.Length; ++step)
            Assert.Equal(secondFromClone[step], secondFromCloneAgain[step]);
        Assert.Equal(rawQsaChat2, RawQsa(model).Bytes);

        // Cold references: chat 2's prompt from nothing, and chat 1's continuation.
        cold.ForwardRefill(shared);
        float[][] secondCold = Continue(cold, second, 3);
        for (int step = 0; step < secondCold.Length; ++step)
            Assert.Equal(secondCold[step], secondFromClone[step]);
        Assert.Equal(RawQsa(cold).Bytes, rawQsaChat2);
        cold.ResetKVCache();
        cold.ForwardRefill(shared);
        float[][] firstCold = Continue(cold, first, 2);
        for (int step = 0; step < firstCold.Length; ++step)
            Assert.Equal(firstCold[step], firstContinuation[step]);

        // Chat 1 itself was not disturbed by two clones being taken beside it.
        Assert.False(model.BindSequenceCache("chat-1"));
        float[] chat1Next = (float[])model.Forward([181]).Clone();
        Assert.Equal(chat1Next, cold.Forward([181]));

        foreach (string id in new[] { "chat-1", "chat-2", "chat-3" }) model.OnSequenceReleased(id);
        Assert.Equal(1, model.RetainedCacheCount);
        model.DiscardRetainedCache("ckpt");
        Assert.Equal(0, model.RetainedCacheCount);
        model.RestorePrimaryCache();
        output.WriteLine($"checkpoint of {shared.Length} tokens cloned twice; clones equal a cold prefill for 3 steps, draft-head state cloned, source unchanged.");
    }

    [Qwen4ExpMtpTinyFact]
    public void DraftHead_SpeculationEngagesOnAReboundHolder_AndMatchesPlainGreedy()
    {
        using var fixture = new Fixture(output);
        using var model = fixture.Load();
        using var cold = fixture.Load();
        const int firstTurn = 8, secondTurn = 12;
        var decoder = new SpeculativeDecoder(model, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead,
            MaxDraftTokens = 3, MaxDraftTokensExplicit = true, MinDraftProb = 0f,
        }) { AdaptiveSpeculation = false, PrefillChunkSize = Prompt.Length };

        // Turn one speculates on holder A, whose draft-head K/V is private to it.
        Assert.True(model.BindSequenceCache("a"));
        var out1 = decoder.GenerateGreedy(Prompt, firstTurn, isStopToken: _ => false);
        Assert.Equal(firstTurn, out1.Count);
        Assert.True(decoder.TokensDrafted > 0 && decoder.VerifySteps > 0);
        object ownerA = Field<Tensor[]>(model, "_gdnConvStateT");
        Assert.True(model.RetainSequenceCache("a"));

        // Another conversation drafts in between, on its own private state (a
        // fresh head starts at its own position 0, whatever the trunk holds).
        Assert.True(model.BindSequenceCache("b"));
        model.ForwardRefill(Other);
        Draft(model, 139, Hidden(model.SpecFeatureSize, 5), 0);

        // The follow-up turn continues the rebound holder with the SAME speculative
        // execution the first turn left (the engine's per-request context), so the
        // head's stashed catch-up rows replay onto the state that travelled with
        // the holder, and the turn drafts and verifies again.
        Assert.True(model.TryRebindRetainedCache("a", "a2"));
        Assert.False(model.BindSequenceCache("a2"));
        Assert.Same(ownerA, Field<Tensor[]>(model, "_gdnConvStateT"));
        var states = Field<IDictionary>(model, "_mtpStates");
        Assert.True(states.Contains(ownerA), "the draft head's private state did not travel with the retained holder");
        int position = model.CacheSeqLen;
        long drafted = decoder.TokensDrafted, verified = decoder.VerifySteps;
        float[] logits = decoder.Prefill(Suffix);
        var out2 = decoder.GenerateGreedyFrom(logits, model.CacheSeqLen, secondTurn, isStopToken: _ => false);
        Assert.Equal(secondTurn, out2.Count);
        Assert.True(decoder.TokensDrafted > drafted, "no tokens were drafted on the rebound holder");
        Assert.True(decoder.VerifySteps > verified, "no verify step ran on the rebound holder");
        output.WriteLine($"turn1={string.Join(',', out1)} turn2={string.Join(',', out2)} drafted={decoder.TokensDrafted} " +
            $"accepted={decoder.TokensAccepted} verify={decoder.VerifySteps} rollback={decoder.RollbackSteps} plain={decoder.PlainSteps}");

        // Plain greedy, uninterrupted, on a model that never speculated. The trunk
        // holds the emitted tokens up to `position`; align the reference to it.
        var expected1 = Greedy(cold, cold.ForwardRefill(Prompt), firstTurn);
        Assert.Equal(expected1, out1);
        Assert.True(position >= cold.CacheSeqLen && position <= Prompt.Length + firstTurn, $"rebound holder holds {position} tokens");
        while (cold.CacheSeqLen < position) cold.Forward([out1[cold.CacheSeqLen - Prompt.Length]]);
        var expected2 = Greedy(cold, cold.Forward(Suffix), secondTurn);
        Assert.Equal(expected2, out2);

        model.OnSequenceReleased("a2");
        model.OnSequenceReleased("b");
        model.RestorePrimaryCache();
    }

    [Qwen4ExpMtpTinyFact]
    public void RetentionBudget_EvictsTheOldestConversation_AndZeroDeclines()
    {
        using var fixture = new Fixture(output);
        long holderBytes;
        using (var probe = fixture.Load())
        {
            Assert.True(probe.BindSequenceCache("probe"));
            // Bound the number of real holders needed for a 1 MB budget; other
            // fixtures exercise growth from the required initial capacity of 8.
            probe.SpecEnsureCapacity(128);
            probe.ForwardRefill(Prompt);
            Assert.True(probe.RetainSequenceCache("probe"));
            holderBytes = probe.RetainedCacheBytes("probe");
            Assert.True(holderBytes > 0);
            probe.DiscardRetainedCache("probe");
            probe.RestorePrimaryCache();
        }
        const long mb = 1024 * 1024;
        long budgetMb = Math.Max(1, (holderBytes + mb - 1) / mb);
        int capacity = (int)(budgetMb * mb / holderBytes);
        Assert.True(capacity >= 1 && capacity <= 32, $"unexpected retention capacity {capacity}");
        output.WriteLine($"holder={holderBytes} bytes budget={budgetMb} MB capacity={capacity} holders");

        string previous = Environment.GetEnvironmentVariable("TS_Q4E_RETAINED_CACHE_MB");
        try
        {
            Environment.SetEnvironmentVariable("TS_Q4E_RETAINED_CACHE_MB", budgetMb.ToString());
            using (var model = fixture.Load())
            {
                for (int i = 0; i <= capacity; ++i)
                {
                    string id = $"conv-{i}";
                    Assert.True(model.BindSequenceCache(id));
                    model.SpecEnsureCapacity(128);
                    model.ForwardRefill(Prompt.Select(t => t + i).ToArray());
                    Assert.True(model.RetainSequenceCache(id), $"{id} was not retained");
                }
                // One more than fit: the oldest conversation was evicted for it, the
                // rest stayed, and the scheduler's later query for the victim declines.
                Assert.Equal(capacity, model.RetainedCacheCount);
                Assert.False(model.CanReuseRetainedPrefix("conv-0", Prompt.Length, Prompt.Length));
                Assert.False(model.TryRebindRetainedCache("conv-0", "conv-0-again"));
                for (int i = 1; i <= capacity; ++i)
                    Assert.True(model.CanReuseRetainedPrefix($"conv-{i}", Prompt.Length, Prompt.Length));
                // The survivor still continues exactly.
                Assert.True(model.TryRebindRetainedCache($"conv-{capacity}", "follow"));
                Assert.False(model.BindSequenceCache("follow"));
                float[] actual = (float[])model.Forward(Suffix).Clone();
                model.ResetKVCache();
                model.ForwardRefill(Prompt.Select(t => t + capacity).ToArray());
                Assert.Equal(actual, model.Forward(Suffix));
                model.OnSequenceReleased("follow");
                model.RestorePrimaryCache();
            }

            Environment.SetEnvironmentVariable("TS_Q4E_RETAINED_CACHE_MB", "0");
            using (var model = fixture.Load())
            {
                Assert.True(model.SupportsRetainedFusedCache);
                Assert.True(model.BindSequenceCache("declined"));
                model.ForwardRefill(Prompt);
                Assert.False(model.RetainSequenceCache("declined"));
                Assert.True(model.HasFusedSequenceCache("declined"));
                Assert.False(model.TryCheckpointActiveCache("ckpt"));
                model.OnSequenceReleased("declined");
                Assert.False(model.HasFusedSequenceCache("declined"));
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_Q4E_RETAINED_CACHE_MB", previous);
        }
    }

    [Qwen4ExpMtpTinyFact]
    public void TeacherForcedTargetVerify_MatchesSingleTokenFullVocabularyForDenseAndSparsePrefixes()
    {
        using var fixture = new Fixture(output);
        using var plain = fixture.Load();
        using var verify = fixture.Load();
        var failures = new List<string>();
        int[] tokens = [151, 157, 163, 167];
        foreach (int[] prefix in new[] { Prompt[..4], Prompt.Concat(Other).ToArray() })
        {
            plain.ResetKVCache();
            verify.ResetKVCache();
            Assert.Equal(plain.ForwardRefill(prefix), verify.ForwardRefill(prefix));
            var expected = tokens.Select(token => (float[])plain.Forward([token]).Clone()).ToArray();
            var actual = new float[tokens.Length * verify.Config.VocabSize];
            verify.SpecForward(tokens, new float[tokens.Length * verify.SpecFeatureSize], actual, allLogitsRows: true);
            for (int row = 0; row < tokens.Length; ++row)
            {
                float[] logits = actual.AsSpan(row * verify.Config.VocabSize, verify.Config.VocabSize).ToArray();
                double max = expected[row].Zip(logits, (x, y) => Math.Abs((double)x - y)).Max();
                output.WriteLine($"teacher prefix={prefix.Length} row={row} max_abs={max:G17} plain={ArgMax(expected[row])} verify={ArgMax(logits)}");
                if (!expected[row].SequenceEqual(logits)) failures.Add($"prefix={prefix.Length} row={row} max_abs={max:G17}");
            }
            ReportChunkStateDifference(verify, plain);
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Qwen4ExpMtpTinyFact]
    public void RepeatedTargetBlocks_MatchScalarTeacherForcingAtEveryCommittedRow()
    {
        using var fixture = new Fixture(output);
        using var scalar = fixture.Load();
        using var blocked = fixture.Load();
        int[] tokens = Enumerable.Range(0, 32).Select(i => 33 + (i * 37 % 210)).ToArray();
        var failures = new List<string>();
        foreach (int[] prefix in new[] { Prompt[..4], Prompt.Concat(Other).ToArray() })
        {
            scalar.ResetKVCache();
            float[] prefixLogits = (float[])scalar.ForwardRefill(prefix).Clone();
            float[][] expected = tokens.Select(token => (float[])scalar.Forward([token]).Clone()).ToArray();
            WriteTeacherVectors($"prefix{prefix.Length}-scalar", expected.SelectMany(row => row).ToArray());
            foreach (int width in new[] { 1, 2, 3, 4 })
            {
                blocked.ResetKVCache();
                Assert.Equal(prefixLogits, blocked.ForwardRefill(prefix));
                int changedRows = 0, changedArgmax = 0;
                double maximum = 0;
                var committed = new List<float>();
                for (int start = 0; start < tokens.Length; start += width)
                {
                    int count = Math.Min(width, tokens.Length - start);
                    var actual = new float[count * blocked.Config.VocabSize];
                    blocked.SpecForward(tokens.AsSpan(start, count).ToArray(),
                        new float[count * blocked.SpecFeatureSize], actual, allLogitsRows: true);
                    committed.AddRange(actual);
                    Assert.Equal(prefix.Length + start + count, blocked.CacheSeqLen);
                    for (int row = 0; row < count; ++row)
                    {
                        float[] got = actual.AsSpan(row * blocked.Config.VocabSize, blocked.Config.VocabSize).ToArray();
                        float[] reference = expected[start + row];
                        Assert.All(got, value => Assert.True(float.IsFinite(value)));
                        double maximumRow = reference.Zip(got, (a, b) => Math.Abs((double)a - b)).Max();
                        maximum = Math.Max(maximum, maximumRow);
                        if (!reference.SequenceEqual(got)) ++changedRows;
                        if (ArgMax(reference) != ArgMax(got)) ++changedArgmax;
                    }
                }
                output.WriteLine($"committed prefix={prefix.Length} width={width} rows={tokens.Length} changed={changedRows} argmax_changed={changedArgmax} max_abs={maximum:G17}");
                WriteTeacherVectors($"prefix{prefix.Length}-width{width}", committed.ToArray());
                ReportChunkStateDifference(blocked, scalar);
                if (changedRows != 0)
                    failures.Add($"prefix={prefix.Length} width={width} changed_rows={changedRows} max_abs={maximum:G17}");
            }
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    private static void WriteTeacherVectors(string name, float[] values)
    {
        string? directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_LOGIT_DUMP_DIR");
        if (string.IsNullOrEmpty(directory)) return;
        Directory.CreateDirectory(directory);
        using var file = new FileStream(Path.Combine(directory, name + ".f32"), FileMode.CreateNew, FileAccess.Write);
        file.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    }

    [Qwen4ExpMtpTinyFact]
    public void SharedPrefixChunking_MatchesWholePromptForEachDistinctSuffix()
    {
        using var fixture = new Fixture(output);
        int[] prefix = Other.Concat(Prompt).ToArray();
        foreach (bool preGrow in new[] { false, true })
        {
            using var chunked = fixture.Load();
            using var whole = fixture.Load();
            if (preGrow)
            {
                chunked.SpecEnsureCapacity(32);
                whole.SpecEnsureCapacity(32);
            }
            foreach (int[] suffix in new[] { new[] { 151, 157, 163, 167 }, new[] { 173, 179, 181, 191 }, new[] { 193, 197, 199, 211 } })
            {
                chunked.ResetKVCache();
                whole.ResetKVCache();
                chunked.ForwardRefill(prefix);
                float[] actual = (float[])chunked.Forward(suffix).Clone();
                float[] expected = (float[])whole.ForwardRefill(prefix.Concat(suffix).ToArray()).Clone();
                // These buffers used to miss native persistent binding because
                // each was below the small-weight threshold. A shape change then
                // uploaded stale host seeds and silently lost the prefix KV.
                foreach (var tensor in Field<Tensor[]>(chunked, "_kCache").Where(t => t != null))
                    Assert.InRange(tensor.Storage.ByteLength, 1, 4095);
                output.WriteLine($"preGrow={preGrow} suffix={suffix[0]} chunked={ArgMax(actual)} whole={ArgMax(expected)}");
                byte[] expectedKeys = RawQsa(whole).Bytes, actualKeys = RawQsa(chunked).Bytes;
                float[] expectedRaw = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, System.Half>(expectedKeys).ToArray().Select(x => (float)x).ToArray();
                float[] actualRaw = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, System.Half>(actualKeys).ToArray().Select(x => (float)x).ToArray();
                output.WriteLine($"raw-key max abs={expectedRaw.Zip(actualRaw, (x, y) => Math.Abs((double)x - y)).Max():G17}; "
                    + $"logit max abs={expected.Zip(actual, (x, y) => Math.Abs((double)x - y)).Max():G17}");
                ReportChunkStateDifference(chunked, whole);
                Assert.Equal(expectedKeys, actualKeys);
                Assert.Equal(expected, actual);
            }
        }
    }

    private unsafe void ReportChunkStateDifference(Qwen4ExpModel chunked, Qwen4ExpModel whole)
    {
        // Diagnostic downloads identify the first stored state that differs. They
        // do not alter the exact cache/logit assertions or native arithmetic.
        var expected = ReadState(whole);
        var actual = ReadState(chunked);
        foreach (string name in expected.Keys)
        {
            var differences = expected[name].Zip(actual[name], (x, y) => Math.Abs((double)x - y)).ToArray();
            output.WriteLine($"state={name} max_abs={differences.Max():G17} changed={differences.Count(x => x != 0)}/{differences.Length}");
        }

        static Dictionary<string, float[]> ReadState(Qwen4ExpModel model)
        {
            typeof(Qwen4ExpModel).GetMethod("EnsureKvCacheHostSynchronized", BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(model, null);
            var result = new Dictionary<string, float[]>();
            Tensor conv = Field<Tensor[]>(model, "_gdnConvStateT")[0];
            Tensor delta = Field<Tensor[]>(model, "_gdnStateT")[0];
            long offset = (conv.Storage.ByteLength + 255) & ~255L;
            byte[] recurrent = Export(TensorComputePrimitives.GetStoragePointer(conv), offset + delta.Storage.ByteLength, model.DeviceForLayer(0));
            result["gdn_conv"] = MemoryMarshal.Cast<byte, float>(recurrent.AsSpan(0, (int)conv.Storage.ByteLength)).ToArray();
            result["gdn_delta"] = MemoryMarshal.Cast<byte, float>(recurrent.AsSpan((int)offset, (int)delta.Storage.ByteLength)).ToArray();
            float[] ple = Field<float[]>(model, "_pleConvState");
            result["ple_conv"] = MemoryMarshal.Cast<byte, float>(Export(Marshal.UnsafeAddrOfPinnedArrayElement(ple, 0), (long)ple.Length * sizeof(float), model.DeviceForLayer(0))).ToArray();
            foreach (string name in new[] { "_kCache", "_vCache" })
            {
                Tensor tensor = Assert.Single(Field<Tensor[]>(model, name).Where(t => t != null));
                Assert.Equal(DType.Float16, tensor.ElementType);
                var bytes = new byte[checked((int)tensor.Storage.ByteLength)];
                Marshal.Copy(TensorComputePrimitives.GetStoragePointer(tensor), bytes, 0, bytes.Length);
                System.Half[] rows = MemoryMarshal.Cast<byte, System.Half>(bytes).ToArray();
                var live = new List<float>();
                for (int head = 0; head < model.Config.NumKVHeads; ++head)
                    for (int row = 0; row < model.CacheSeqLen; ++row)
                        for (int dim = 0; dim < model.Config.HeadDim; ++dim)
                            live.Add((float)rows[(head * Field<int>(model, "_kvCacheCapacity") + row) * model.Config.HeadDim + dim]);
                result[name] = live.ToArray();
            }
            return result;
        }

        static byte[] Export(IntPtr key, long length, int device)
        {
            var result = new byte[checked((int)length)];
            fixed (byte* destination = result)
                Assert.True(GgmlBasicOps.Qwen4ExpCopyQsaCache(key, (IntPtr)destination, length, device));
            return result;
        }
    }

    [Qwen4ExpMtpTinyFact]
    public async Task Engine_FollowUpReusesRetainedHolder_AndNewChatsCloneTheCheckpoint()
    {
        string[] keys = { "TS_RETAINED_FUSED_CACHE", "TS_PER_SEQ_FUSED", "TS_PREFIX_CHECKPOINTS", "TS_PREFIX_CHECKPOINTS_MAX" };
        string[] values = { "1", "1", "1", "2" };
        var old = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        try
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], values[i]);
            using var fixture = new Fixture(output);
            using var model = fixture.Load();
            using var cold = fixture.Load();
            const int blockSize = 8, newTokens = 6;
            var config = new SchedulerConfig
            {
                MaxNumBatchedTokens = 1024, MaxNumRunningSequences = 8,
                MaxPrefillChunkSize = 256, SoloPrefillChunkSize = 256,
                NumBlocks = 256, BlockSize = blockSize, EnablePrefixCaching = true,
                DecodeQuantumTokens = 1,
            };
            using var engine = new InferenceEngine(model, config, NullLogger.Instance);

            // Round one: two concurrent conversations, each in its own fused holder.
            var promptA = Prompt.Concat(Media).ToList();
            var promptB = Other.Concat(AfterMedia).ToList();
            var handleA = engine.SubmitRequest(new SequenceState("A1", promptA, newTokens, blockSize, SamplingConfig.Greedy));
            var handleB = engine.SubmitRequest(new SequenceState("B1", promptB, newTokens, blockSize, SamplingConfig.Greedy));
            var drainA = DrainAsync(handleA);
            var drainB = DrainAsync(handleB);
            await Task.WhenAll(drainA, drainB);
            var (completionA1, outA) = await drainA;
            var (completionB1, outB) = await drainB;
            Assert.Equal(0, completionA1.PrefixCacheReusedTokens);
            Assert.Equal(0, completionB1.PrefixCacheReusedTokens);
            Assert.True(outA.Count > 0 && outB.Count > 0);

            // The follow-up turn of A extends A's retained holder exactly and pays
            // only for its suffix. Its tokens are what an uninterrupted plain
            // continuation gives.
            var followA = promptA.Concat(outA).Concat(Suffix).ToList();
            var (completionA2, outA2) = await DrainAsync(engine.SubmitRequest(
                new SequenceState("A2", followA, newTokens, blockSize, SamplingConfig.Greedy)));
            output.WriteLine($"A1 out={string.Join(',', outA)} A2 reused={completionA2.PrefixCacheReusedTokens} of {followA.Count} out={string.Join(',', outA2)}");
            Assert.True(completionA2.PrefixCacheReusedTokens >= promptA.Count + outA.Count - 1,
                $"follow-up reused {completionA2.PrefixCacheReusedTokens} tokens; expected the retained conversation ({promptA.Count + outA.Count - 1}+)");
            Assert.True(completionA2.PrefixCacheReusedTokens <= followA.Count - 1);
            Assert.True(outA2.Count > 0);
            var expectedA2 = Greedy(cold, cold.ForwardRefill(followA.ToArray()), outA2.Count);
            Assert.Equal(expectedA2, outA2);

            // New chats sharing a prefix: the first one is checkpointed at the
            // boundary, every later one starts from a clone of it.
            var shared = Other.Concat(Prompt).ToList();
            var chat1 = shared.Concat(new[] { 151, 157, 163, 167 }).ToList();
            var chat2 = shared.Concat(new[] { 173, 179, 181, 191 }).ToList();
            var chat3 = shared.Concat(new[] { 193, 197, 199, 211 }).ToList();
            var (c1, out1) = await DrainAsync(engine.SubmitRequest(
                new SequenceState("chat-1", chat1, newTokens, blockSize, SamplingConfig.Greedy, sharedPrefixTokens: shared.Count)));
            Assert.Equal(0, c1.PrefixCacheReusedTokens);
            Assert.True(model.RetainedCacheCount >= 1, "no checkpoint was taken at the shared-prefix boundary");
            var (c2, out2) = await DrainAsync(engine.SubmitRequest(
                new SequenceState("chat-2", chat2, newTokens, blockSize, SamplingConfig.Greedy, sharedPrefixTokens: shared.Count)));
            var (c3, out3) = await DrainAsync(engine.SubmitRequest(
                new SequenceState("chat-3", chat3, newTokens, blockSize, SamplingConfig.Greedy, sharedPrefixTokens: shared.Count)));
            output.WriteLine($"chat-1 out={string.Join(',', out1)} chat-2 reused={c2.PrefixCacheReusedTokens} out={string.Join(',', out2)} chat-3 reused={c3.PrefixCacheReusedTokens} out={string.Join(',', out3)}");
            Assert.Equal(shared.Count, c2.PrefixCacheReusedTokens);
            Assert.Equal(shared.Count, c3.PrefixCacheReusedTokens);
            Assert.True(out2.Count > 0 && out3.Count > 0);
            cold.ResetKVCache();
            Assert.Equal(Greedy(cold, cold.ForwardRefill(chat2.ToArray()), out2.Count), out2);
            cold.ResetKVCache();
            Assert.Equal(Greedy(cold, cold.ForwardRefill(chat3.ToArray()), out3.Count), out3);
            cold.ResetKVCache();
            Assert.Equal(Greedy(cold, cold.ForwardRefill(chat1.ToArray()), out1.Count), out1);
        }
        finally
        {
            for (int i = 0; i < keys.Length; ++i) Environment.SetEnvironmentVariable(keys[i], old[i]);
        }
    }

    // ---- helpers ----

    private static void RunConversation(Qwen4ExpModel model)
    {
        model.ForwardRefill(Prompt);
        model.SetMRoPEPositions((int[])MediaAxes.Clone());
        model.Forward(Media);
        model.SetMRoPEPositions((int[])AfterMediaAxes.Clone());
        model.Forward(AfterMedia);
    }

    /// <summary>Forward <paramref name="tokens"/>, then <paramref name="steps"/> greedy
    /// decode steps; every step's logits, cloned.</summary>
    private static float[][] Continue(Qwen4ExpModel model, int[] tokens, int steps)
    {
        var result = new float[steps + 1][];
        float[] logits = model.Forward(tokens);
        result[0] = (float[])logits.Clone();
        for (int step = 1; step <= steps; ++step)
        {
            logits = model.Forward([ArgMax(logits)]);
            result[step] = (float[])logits.Clone();
        }
        return result;
    }

    private static List<int> Greedy(Qwen4ExpModel model, float[] logits, int count)
    {
        var tokens = new List<int>(count);
        for (int i = 0; i < count; ++i)
        {
            int token = ArgMax(logits);
            tokens.Add(token);
            if (i + 1 < count) logits = model.Forward([token]);
        }
        return tokens;
    }

    private static async Task<(InferenceCompletion completion, List<int> output)> DrainAsync(InferenceRequestHandle handle)
    {
        var tokens = new List<int>();
        await foreach (int t in handle.Tokens.ReadAllAsync())
            tokens.Add(t);
        var completion = await handle.Completion;
        return (completion, tokens);
    }

    private static (float[] Logits, float[] Hidden) Draft(Qwen4ExpModel model, int token, float[] hidden, int pos)
    {
        var logits = new float[model.Config.VocabSize];
        var next = new float[model.SpecFeatureSize];
        model.DraftStep(token, hidden, pos, logits, next);
        return (logits, next);
    }

    private static float[] Hidden(int size, int seed) => Enumerable.Range(0, size)
        .Select(i => (float)Math.Sin((i + seed) * .37)).ToArray();

    private static T Field<T>(Qwen4ExpModel model, string name) => (T)typeof(Qwen4ExpModel)
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static Tensor QsaTensor(Qwen4ExpModel model)
        => Assert.Single(Field<Tensor[]>(model, "_idxKCache").Where(t => t != null));

    /// <summary>The ACTIVE holder's raw indexer-key bytes for its written rows, read
    /// from the authoritative native entry.</summary>
    private static unsafe (IntPtr Key, byte[] Bytes) RawQsa(Qwen4ExpModel model)
    {
        Tensor tensor = QsaTensor(model);
        IntPtr key = TensorComputePrimitives.GetStoragePointer(tensor);
        var bytes = new byte[checked((int)tensor.Storage.ByteLength)];
        fixed (byte* destination = bytes)
            Assert.True(GgmlBasicOps.Qwen4ExpCopyQsaCache(key, (IntPtr)destination, bytes.Length, model.DeviceForLayer(Array.FindIndex(Field<Tensor[]>(model, "_idxKCache"), t => t != null))));
        int capacity = Field<int>(model, "_kvCacheCapacity");
        int rowBytes = bytes.Length / capacity;
        return (key, bytes[..checked(rowBytes * model.CacheSeqLen)]);
    }

    private static int ArgMax(float[] row)
    {
        Assert.All(row, value => Assert.True(float.IsFinite(value)));
        int best = 0;
        for (int i = 1; i < row.Length; i++) if (row[i] > row[best]) best = i;
        return best;
    }

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
                Assert.Equal(manifest.RootElement.GetProperty("files").GetProperty(name).GetProperty("sha256").GetString(),
                    Hash(Path.Combine(_path, name)));
        }
        internal Qwen4ExpModel Load(int layerSplitDegree = 1)
        {
            var backend = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MTP_BACKEND") switch
            {
                null or "" or "GgmlCpu" => BackendType.GgmlCpu,
                "GgmlCuda" => BackendType.GgmlCuda,
                "GgmlMetal" => BackendType.GgmlMetal,
                var unsupported => throw new InvalidOperationException($"Unsupported fixture backend {unsupported}"),
            };
            var model = new Qwen4ExpModel(Path.Combine(_path, "target.gguf"), backend,
                layerSplitDegree: layerSplitDegree, draftGgufPath: Path.Combine(_path, "head.gguf"));
            try
            {
                string native = TestGates.MappedNativeGgmlOpsPath();
                string hash = Hash(native);
                Assert.Equal(Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_NATIVE_SHA256"), hash);
                _output.WriteLine($"native {native} sha256={hash}; backend={backend}");
                Assert.Equal(260, model.Config.VocabSize);
                Assert.True(model.HasDraftHead);
                return model;
            }
            catch { model.Dispose(); throw; }
        }
        private static bool IsNativeLibrary(string path)
        {
            string name = Path.GetFileName(path);
            return string.Equals(name, "GgmlOps.dll", StringComparison.OrdinalIgnoreCase)
                || string.Equals(name, "libGgmlOps.so", StringComparison.OrdinalIgnoreCase)
                || string.Equals(name, "libGgmlOps.dylib", StringComparison.OrdinalIgnoreCase);
        }
        private static string Hash(string path) => Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path)));
        public void Dispose() { }
    }
}
