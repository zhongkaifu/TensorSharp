// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// The drafter's context belongs to its sequence (issue #376): a block drafter's ring keeps a finished request's rows, the
/// next request whose tokens begin with them continues it, a request that does not speculate still feeds its prompt into a
/// ring (one no running request holds), and a ring a request takes anew holds nothing of another sequence. Two models attach
/// the same DFlash2 drafter (window 16, a 32-row ring) to a synthetic Qwen 3.5 dense trunk.
/// </summary>
public sealed class DraftContextContinuationTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-draft-context-" + Guid.NewGuid().ToString("N"));
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public DraftContextContinuationTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // ---------------------------------------------------------------- the runtime's registry

    [Fact]
    public void TheRegistryOffersTheLongestUsablePrefixFirstAndTheNewestOnATie()
    {
        var registry = new DraftContextRegistry();
        int[] tokens = Enumerable.Range(0, 50).ToArray();
        Assert.Null(registry.Note("short", tokens, 20, Array.Empty<PromptMediaSpan>(), null));
        Assert.Null(registry.Note("long", tokens, 40, Array.Empty<PromptMediaSpan>(), null));
        Assert.Null(registry.Note("tie", tokens, 40, Array.Empty<PromptMediaSpan>(), null));
        Assert.Null(registry.Note("none", tokens, 5, Array.Empty<PromptMediaSpan>(), null));
        // A sequence whose first 30 tokens are these, then others: each context's usable prefix is its common prefix.
        Func<int, int> sequence = p => p < 30 ? p : -1;
        var candidates = registry.Candidates(kept => kept.Key == "none" ? 0 : DraftContextRegistry.CommonPrefix(kept, sequence, 45));
        Assert.Equal(new[] { ("tie", 30), ("long", 30), ("short", 20) }, candidates);
        Assert.True(registry.Forget("tie"));
        Assert.False(registry.Forget("tie"));
        Assert.Equal(3, registry.Count);
    }

    [Fact]
    public void TheRegistryDropsItsOldestContextPastItsCapacity()
    {
        var registry = new DraftContextRegistry();
        int[] tokens = { 1, 2, 3 };
        for (int i = 0; i < DraftContextRegistry.Capacity; i++)
            Assert.Null(registry.Note($"k{i}", tokens, 3, Array.Empty<PromptMediaSpan>(), null));
        Assert.Equal("k0", registry.Note("last", tokens, 3, Array.Empty<PromptMediaSpan>(), null));
        // Kept again under the same key, an entry replaces its earlier one.
        Assert.Null(registry.Note("last", tokens, 2, Array.Empty<PromptMediaSpan>(), null));
        Assert.Equal(DraftContextRegistry.Capacity, registry.Count);
    }

    // ---------------------------------------------------------------- the model's rings

    /// <summary>
    /// A request that does not speculate feeds its prompt into a ring; kept at its end, the ring is the context of the request
    /// that continues it: the rows and the carry a speculating request's followed prompt leaves, bit for bit. Continued over
    /// fewer positions, the rows past them (and those whose slots positions past them took) are not context, and the carry
    /// is not held. A request that takes a ring anew sees none of another sequence's rows.
    /// </summary>
    [Fact]
    public void AKeptRingIsTheContextOfTheRequestThatContinuesIt() => KeptRingContinues(BackendType.Cpu);

    [CudaFact]
    public void AKeptRingIsTheContextOfTheRequestThatContinuesItOnDirectCuda() => KeptRingContinues(BackendType.Cuda);

    private void KeptRingContinues(BackendType backend)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        using var aside = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        using var reference = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        string label = $"{backend}";
        Assert.True(((IDraftHead)aside).DraftSelfCatchUp, $"{label}: the trunk does not keep the ring");
        int ring = aside.DFlashRingRowsForTests, feat = ((ISpeculativeModel)aside).SpecFeatureSize;
        int[] prompt = Enumerable.Range(0, 40).Select(i => 40 + (i * 7) % 23).ToArray();
        int n = prompt.Length;
        Assert.True(n > ring);

        // The same chunks: one request that does not speculate, one that does.
        Prefill(aside, "a", prompt, speculates: false);
        Prefill(reference, "x", prompt, speculates: true);
        Assert.True(aside.GraphDFlashFollows > 0, $"{label}: the prompt of a request that does not speculate reached no ring");
        Assert.Equal(reference.GraphDFlashFollows, aside.GraphDFlashFollows);

        var states = (IPerRequestDraftState)aside;
        Assert.Equal(n, states.KeepDraftState("a"));
        Assert.True(states.ContinueDraftState("c", "a", n));
        Assert.False(states.ContinueDraftState("c2", "a", n), $"{label}: a kept context was continued twice");
        var mismatches = new List<string>();
        CompareContexts(aside, "c", reference, "x", label + " continued", mismatches, from: n - ring, to: n);
        var (carry, expected) = (new float[feat], new float[feat]);
        states.BindDraftState("c");
        ((IPerRequestDraftState)reference).BindDraftState("x");
        Assert.True(((IDraftHead)aside).TryTakeHeldCarry(n, carry), $"{label}: the continued ring holds no carry");
        Assert.True(((IDraftHead)reference).TryTakeHeldCarry(n, expected));
        Assert.True(carry.AsSpan().SequenceEqual(expected), $"{label}: the carry differs");

        // A request that takes a ring anew: nothing of another sequence's.
        states.BindDraftState("e");
        Assert.Equal((0, 0), aside.DFlashRingRunForTests("e"));

        // Continued over fewer positions: the rows before them only, past those a later position's slot took.
        Assert.Equal(n, states.KeepDraftState("c"));
        int fewer = 30, block = aside.DFlashSettings.BlockSize;
        Assert.True(states.ContinueDraftState("d", "c", fewer));
        int from = Math.Max(n - ring, n + block - ring);
        Assert.Equal((from, fewer), aside.DFlashRingRunForTests("d"));
        CompareContexts(aside, "d", reference, "x", label + " fewer", mismatches, from, fewer);
        states.BindDraftState("d");
        Assert.False(((IDraftHead)aside).TryTakeHeldCarry(fewer, carry), $"{label}: a carry held past the continued positions");
        Assert.True(mismatches.Count == 0, $"{label}: {string.Join("; ", mismatches.Take(12))}");
        Assert.Equal(2, aside.DFlashRingsContinued);
    }

    /// <summary>A prompt of a request that does not speculate takes a free ring only: with every ring held by a running
    /// request, it reaches none and theirs keep their rows.</summary>
    [Fact]
    public void APromptThatDoesNotSpeculateNeverTakesARunningRequestsRing()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        var states = (IPerRequestDraftState)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 41 + (i * 5) % 19).ToArray();
        // The model's own ring and eight pooled ones, each holding a speculating request's prompt.
        for (int r = 0; r < 9; r++)
            Prefill(model, $"s{r}", prompt, speculates: true);
        Assert.Equal(9, model.DFlashRingCountForTests);
        int follows = model.GraphDFlashFollows;
        Prefill(model, "z", prompt, speculates: false);
        Assert.Equal(follows, model.GraphDFlashFollows);
        Assert.Null(model.DFlashRingRunForTests("z"));
        for (int r = 0; r < 9; r++)
            Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests($"s{r}"));
        // One ends and keeps its context: still no free ring, the kept context is taken before a running request's.
        Assert.Equal(prompt.Length, states.KeepDraftState("s0"));
        Prefill(model, "y", prompt, speculates: false);
        Assert.Equal(follows + (prompt.Length + 15) / 16, model.GraphDFlashFollows);
        Assert.Null(model.DFlashRingRunForTests("s0"));
        for (int r = 1; r < 9; r++)
            Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests($"s{r}"));
    }

    /// <summary>
    /// Device memory short with no conversation the prefix cache keeps left to release: the rings finished requests keep go
    /// (with their contexts), as many as the room needs, and the running requests' rings stay.
    /// </summary>
    [Fact]
    public void KeptRingsGoWhenNoConversationIsLeftToRelease()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        var states = (IPerRequestDraftState)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 41 + (i * 5) % 19).ToArray();
        for (int r = 0; r < 4; r++)
            Prefill(model, $"k{r}", prompt, speculates: true);
        for (int r = 0; r < 3; r++)
            Assert.Equal(prompt.Length, states.KeepDraftState($"k{r}"));
        Assert.Equal(4, model.DFlashRingCountForTests);
        long ringBytes = 2L * model.DFlashSettings.NumLayers * model.DFlashSettings.NumKVHeads * model.DFlashRingRowsForTests
            * model.DFlashSettings.HeadDim * sizeof(float) + model.DFlashSettings.FeatureSize * sizeof(float);
        // The stand-in device frees a ring's bytes for each ring given back; the room asked for is two rings' worth.
        const long gb = 1L << 30;
        model.DeviceRoomForTests = () => ((4 - model.DFlashRingCountForTests) * ringBytes, 64 * gb);
        Assert.True(model.EnsureRecordingRoomForTests(2 * ringBytes));
        // The pooled rings no request holds went (k1, k2: the model's own ring k0 stays, and k3's request runs).
        Assert.Equal(2, model.DFlashRingCountForTests);
        Assert.Equal(1, model.DeviceRoomStages[4]);
        Assert.Equal(0, model.DeviceRoomStages[5]);
        Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests("k3"));
        Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests("k0"));
        Assert.Null(model.DFlashRingRunForTests("k1"));
        Assert.False(states.ContinueDraftState("n", "k1", prompt.Length));
    }

    /// <summary>
    /// Device memory short: a conversation the prefix cache keeps goes before any kept ring, even where the rings would cover
    /// the shortfall (an 81 MB ring is little room beside a conversation and worth its next turn's drafts: given back first, on
    /// a 32 GB RTX PRO 4500 at 131K context, they left reused turns of kept conversations drafting without their context).
    /// With no conversation left, only the least recently kept rings the room needs go. A check that may release no
    /// conversation (a slot's roomier target) gives no ring back either.
    /// </summary>
    [Fact]
    public void AConversationGoesBeforeAnyKeptRing()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        var states = (IPerRequestDraftState)model;
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 41 + (i * 5) % 19).ToArray();
        // k0 on the model's own ring, k1 and k2 on pooled ones, kept in that order.
        for (int r = 0; r < 3; r++)
            Prefill(model, $"k{r}", prompt, speculates: true);
        for (int r = 0; r < 3; r++)
            Assert.Equal(prompt.Length, states.KeepDraftState($"k{r}"));
        // A conversation the prefix cache keeps.
        unified.EnsureSequenceSlot("conv", prompt.Length + 32);
        Assert.True(unified.TryForwardUnified(new[] { "conv" }, new[] { prompt }, new[] { 0 }, new float[1][]));
        Assert.True(model.RetainSequenceCacheAs("conv", "conv"));
        Assert.Equal(3, model.DFlashRingCountForTests);
        long ringBytes = 2L * model.DFlashSettings.NumLayers * model.DFlashSettings.NumKVHeads * model.DFlashRingRowsForTests
            * model.DFlashSettings.HeadDim * sizeof(float) + model.DFlashSettings.FeatureSize * sizeof(float);
        const long gb = 1L << 30;

        // A check that may release no conversation keeps the rings too (a slot's roomier target asks so): no room, nothing gone.
        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count == 0 ? 3 * ringBytes : 0, 64 * gb);
        Assert.False(model.EnsureRecordingRoomForTests(ringBytes, mayEvict: false));
        Assert.Equal(new[] { "conv" }, model.RetainedPayloadKeys);
        Assert.Equal(3, model.DFlashRingCountForTests);
        Assert.Equal(0, model.DeviceRoomStages[4]);

        // One ring's worth short, two kept rings to give: the conversation goes and makes the room, the rings stay.
        model.DeviceRoomForTests = () => (model.RetainedPayloadKeys.Count == 0 ? 3 * ringBytes : 0, 64 * gb);
        Assert.True(model.EnsureRecordingRoomForTests(ringBytes));
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(3, model.DFlashRingCountForTests);
        Assert.Equal(0, model.DeviceRoomStages[4]);
        Assert.Equal(1, model.DeviceRoomStages[5]);

        // One ring's worth short and no conversation left: the ring kept the longest ago goes (k1's), k2's stays.
        model.DeviceRoomForTests = () => ((3 - model.DFlashRingCountForTests) * ringBytes, 64 * gb);
        Assert.True(model.EnsureRecordingRoomForTests(ringBytes));
        Assert.Equal(2, model.DFlashRingCountForTests);
        Assert.Equal(1, model.DeviceRoomStages[4]);
        Assert.Null(model.DFlashRingRunForTests("k1"));
        Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests("k2"));
        Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests("k0"));
    }

    /// <summary>
    /// Making room releases conversations and then kept rings only where the two together reach the need (issue #376's
    /// review: the room check that releases no conversation where releasing them all could not make the room, merged with the
    /// rings given back after the conversations). Short of more than both free, nothing goes - neither the conversation nor
    /// the rings and their contexts; short of what both free together, both go and the room is made, where counting the
    /// conversation alone had kept it and given the rings back for nothing.
    /// </summary>
    [Fact]
    public void ConversationsAndKeptRingsGoOnlyWhereTogetherTheyMakeTheRoom()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        var states = (IPerRequestDraftState)model;
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 41 + (i * 5) % 19).ToArray();
        // k0 on the model's own ring, k1 and k2 on pooled ones (the two a room check may give back), kept in that order.
        for (int r = 0; r < 3; r++)
            Prefill(model, $"k{r}", prompt, speculates: true);
        for (int r = 0; r < 3; r++)
            Assert.Equal(prompt.Length, states.KeepDraftState($"k{r}"));
        unified.EnsureSequenceSlot("conv", prompt.Length + 32);
        Assert.True(unified.TryForwardUnified(new[] { "conv" }, new[] { prompt }, new[] { 0 }, new float[1][]));
        Assert.True(model.RetainSequenceCacheAs("conv", "conv"));
        Assert.Equal(3, model.DFlashRingCountForTests);
        long ringBytes = 2L * model.DFlashSettings.NumLayers * model.DFlashSettings.NumKVHeads * model.DFlashRingRowsForTests
            * model.DFlashSettings.HeadDim * sizeof(float) + model.DFlashSettings.FeatureSize * sizeof(float);
        const long gb = 1L << 30;
        // The conversation frees a ring's worth, each ring given back its own.
        model.RetainedPayloadBytesForTests = _ => ringBytes;
        model.DeviceRoomForTests = () =>
            (((model.RetainedPayloadKeys.Count == 0 ? 1 : 0) + 3 - model.DFlashRingCountForTests) * ringBytes, 64 * gb);

        // Four rings' worth short, three in reach: nothing goes.
        Assert.False(model.EnsureRecordingRoomForTests(4 * ringBytes));
        Assert.Equal(new[] { "conv" }, model.RetainedPayloadKeys);
        Assert.Equal(3, model.DFlashRingCountForTests);
        Assert.Equal(0, model.DeviceRoomStages[4]);
        Assert.Equal(0, model.DeviceRoomStages[5]);
        Assert.Equal(1, model.DeviceRoomReleasesSkipped);

        // Three rings' worth short: the conversation and both idle rings go, and the room is made.
        Assert.True(model.EnsureRecordingRoomForTests(3 * ringBytes));
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(1, model.DFlashRingCountForTests);
        Assert.Equal(1, model.DeviceRoomStages[5]);
        Assert.Equal(1, model.DeviceRoomStages[4]);
        Assert.Equal((0, prompt.Length), model.DFlashRingRunForTests("k0"));
    }

    // ---------------------------------------------------------------- the engine

    /// <summary>
    /// A conversation's turns through the engine with its prefix cache: A (a prompt, one token), B (A and a new turn, one
    /// token), C (B's prompt again, decoding with speculation). A and B do not speculate, yet their prompts feed the ring; B
    /// continues A's context, C continues B's whole prompt with its carry and drafts its first step from it; C's stream is
    /// plain decoding's. Before, C drafted over whatever ring the last speculating request had left.
    /// </summary>
    [Fact]
    public Task AReusedTurnDraftsOverItsConversation() => ReusedTurnDraftsOverItsConversation(BackendType.Cpu, KvCacheDtype.F32);

    [CudaFact]
    public Task AReusedTurnDraftsOverItsConversationOnDirectCuda() => ReusedTurnDraftsOverItsConversation(BackendType.Cuda, KvCacheDtype.F32);

    private async Task ReusedTurnDraftsOverItsConversation(BackendType backend, KvCacheDtype kv)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(kv);
        var (target, draft) = Fixtures();
        int[][] turns = Conversation(0);
        int[][] plain = (await ServeConversation(target, draft, backend, speculate: false, turns)).Outputs;

        using var model = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        using var engine = new InferenceEngine(model, Config(speculate: true), NullLogger.Instance);
        var served = new List<(int[] Tokens, InferenceCompletion Completion, SequenceState Seq)>();
        int carries = 0, follows = 0;
        for (int t = 0; t < turns.Length; t++)
        {
            if (t == 2) (carries, follows) = (model.HeldCarriesTaken, model.GraphDFlashFollows);
            served.Add(await Serve(engine, $"c0t{t}", turns[t], t == 2 ? CMaxNew : 1).WaitAsync(TimeSpan.FromMinutes(3)));
            if (t == 0) Assert.True(model.GraphDFlashFollows > 0, $"{backend}: A's prompt reached no ring");
        }
        var (tokens, completion, seq) = served[2];
        string label = $"{backend}";
        _output.WriteLine($"{label}: C reused {completion.PrefixCacheReusedTokens} of {turns[2].Length}, drafted " +
            $"{seq.SpecStats?.TokensDrafted} accepted {seq.SpecStats?.TokensAccepted} in {seq.SpecStats?.VerifySteps} verify steps, " +
            $"continued {engine.ExecutorForTests.DraftContextsContinued}, kept run {model.DFlashRingRunForTests(seq.RequestId)}");
        Assert.Null(served[0].Seq.SpecStats);
        Assert.Null(served[1].Seq.SpecStats);
        Assert.Equal(turns[0].Length, served[1].Completion.PrefixCacheReusedTokens);
        Assert.Equal(turns[2].Length, completion.PrefixCacheReusedTokens);
        // A's and B's prompts, read with the ring's writes, draw what they draw without a drafter's context to keep.
        Assert.Equal(plain[0], served[0].Tokens);
        Assert.Equal(plain[1], served[1].Tokens);
        Assert.Equal(plain[2], tokens);
        Assert.Equal(2, engine.ExecutorForTests.DraftContextsContinued);
        Assert.True(seq.SpecStats is { TokensDrafted: > 0 }, $"{label}: C did not draft");
        Assert.True(model.HeldCarriesTaken > carries, $"{label}: C's first draft did not chain from the carry B's prompt left");
        Assert.Equal(follows, model.GraphDFlashFollows);   // C read nothing: the context is B's prompt
        // After C the ring C kept holds a whole window of the conversation, not a run restarted at C's first draft.
        var run = model.DFlashRingRunForTests(seq.RequestId);
        Assert.True(run is { } r && r.Filled > turns[2].Length && r.ValidFrom == Math.Max(0, r.Filled - model.DFlashRingRowsForTests)
            && r.ValidFrom < turns[2].Length - 1, $"{label}: C's ring holds {run}");
    }

    /// <summary>
    /// The reused turn takes its conversation's context as it adopts the cached prefix, before the copy of that prefix (or
    /// its slot) asks the device for room, which gives back the contexts no request holds: with two other conversations'
    /// first turns kept (X on the model's own ring, Y and the conversation on pooled ones) and the device short as C starts,
    /// Y's ring goes and C still drafts over B's prompt. On a 32 GB RTX PRO 4500 at 131K context every reused longturn turn
    /// past 8K ran short there and, its context given back, drafted over another conversation's rows (issue #376).
    /// </summary>
    [Fact]
    public async Task AReusedTurnHoldsItsContextBeforeItsCacheAsksForRoom()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        int[][] turns = Conversation(0);
        int[] plain = (await ServeConversation(target, draft, BackendType.Cpu, speculate: false, turns)).Outputs[2];

        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        const long gb = 1L << 30;
        // Room everywhere until C starts; then only the headroom (2 GB of 64: the prefix cache sees no deficit and keeps B's
        // end state for C) while three rings are held - giving one back makes the room.
        bool tight = false;
        model.DeviceRoomForTests = () => (tight && model.DFlashRingCountForTests >= 3 ? 2 * gb : 64 * gb, 64 * gb);
        using var engine = new InferenceEngine(model, Config(speculate: true), NullLogger.Instance);
        await Serve(engine, "c1t0", Conversation(1)[0], 1).WaitAsync(TimeSpan.FromMinutes(3));
        await Serve(engine, "c2t0", Conversation(2)[0], 1).WaitAsync(TimeSpan.FromMinutes(3));
        await Serve(engine, "c0t0", turns[0], 1).WaitAsync(TimeSpan.FromMinutes(3));
        await Serve(engine, "c0t1", turns[1], 1).WaitAsync(TimeSpan.FromMinutes(3));
        Assert.Equal(3, model.DFlashRingCountForTests);
        int stages = model.DeviceRoomStages[4];
        tight = true;
        var (tokens, completion, seq) = await Serve(engine, "c0t2", turns[2], CMaxNew).WaitAsync(TimeSpan.FromMinutes(3));
        _output.WriteLine($"C reused {completion.PrefixCacheReusedTokens} of {turns[2].Length}, drafted {seq.SpecStats?.TokensDrafted} " +
            $"accepted {seq.SpecStats?.TokensAccepted}, continued {engine.ExecutorForTests.DraftContextsContinued}, " +
            $"{model.DFlashRingCountForTests} rings, room stages {string.Join(",", model.DeviceRoomStages)}, kept run {model.DFlashRingRunForTests(seq.RequestId)}");
        // The device ran short as C started, and gave a ring back for it.
        Assert.Equal(2, model.DFlashRingCountForTests);
        Assert.True(model.DeviceRoomStages[4] > stages);
        Assert.Equal(turns[2].Length, completion.PrefixCacheReusedTokens);
        Assert.Equal(plain, tokens);
        // B continued A's context, and C B's: the ring that went was Y's.
        Assert.Equal(2, engine.ExecutorForTests.DraftContextsContinued);
        Assert.Null(model.DFlashRingRunForTests("c2t0"));
        Assert.True(seq.SpecStats is { TokensDrafted: > 0 }, "C did not draft");
        var run = model.DFlashRingRunForTests(seq.RequestId);
        Assert.True(run is { } r && r.Filled > turns[2].Length && r.ValidFrom == Math.Max(0, r.Filled - model.DFlashRingRowsForTests)
            && r.ValidFrom < turns[2].Length - 1, $"C's ring holds {run}");
    }

    /// <summary>
    /// A model that gives no kept ring back when memory runs short (every device but one direct-CUDA device, whose room the
    /// model makes: EnsureDeviceRoom) makes no ring to keep a context beside the kept ones: a request takes the context kept
    /// the longest ago, so conversations served one after another hold one ring between them, as they did before contexts
    /// were kept, and a conversation's next turns still continue its own context. Made beside them, every distinct
    /// conversation had kept a ring of its own up to the pool's nine (85 MB each for Qwen3.8-27B's DFlash2) for the life of
    /// the process on ggml_cuda, ggml_metal, ggml_vulkan, MLX and the CPUs, where nothing gives a ring back.
    /// </summary>
    [Fact]
    public async Task KeptContextsAddNoRingWhereNothingGivesThemBack()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        int[][] turns = Conversation(0);
        int[] plain = (await ServeConversation(target, draft, BackendType.Cpu, speculate: false, turns)).Outputs[2];

        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        using var engine = new InferenceEngine(model, Config(speculate: true), NullLogger.Instance);
        // Two other conversations, one after the other: one speculates, one does not (its prompt still feeds a ring).
        await Serve(engine, "c1t0", Conversation(1)[0], CMaxNew).WaitAsync(TimeSpan.FromMinutes(3));
        await Serve(engine, "c2t0", Conversation(2)[0], 1).WaitAsync(TimeSpan.FromMinutes(3));
        Assert.Equal(1, model.DFlashRingCountForTests);
        var served = new List<(int[] Tokens, InferenceCompletion Completion, SequenceState Seq)>();
        for (int t = 0; t < turns.Length; t++)
            served.Add(await Serve(engine, $"c0t{t}", turns[t], t == 2 ? CMaxNew : 1).WaitAsync(TimeSpan.FromMinutes(3)));
        var (tokens, completion, seq) = served[2];
        _output.WriteLine($"C reused {completion.PrefixCacheReusedTokens} of {turns[2].Length}, drafted {seq.SpecStats?.TokensDrafted} " +
            $"accepted {seq.SpecStats?.TokensAccepted}, continued {engine.ExecutorForTests.DraftContextsContinued}, " +
            $"{model.DFlashRingCountForTests} rings, kept run {model.DFlashRingRunForTests(seq.RequestId)}");
        Assert.Equal(1, model.DFlashRingCountForTests);
        Assert.Equal(turns[2].Length, completion.PrefixCacheReusedTokens);
        Assert.Equal(plain, tokens);
        // B continued A's context, and C B's.
        Assert.Equal(2, engine.ExecutorForTests.DraftContextsContinued);
        Assert.True(seq.SpecStats is { TokensDrafted: > 0 }, "C did not draft");
        var run = model.DFlashRingRunForTests(seq.RequestId);
        Assert.True(run is { } r && r.Filled > turns[2].Length && r.ValidFrom == Math.Max(0, r.Filled - model.DFlashRingRowsForTests)
            && r.ValidFrom < turns[2].Length - 1, $"C's ring holds {run}");
    }

    /// <summary>
    /// Another conversation speculates between a conversation's last prompt and its reused turn (A0, B0, X1, C0, X1 a fresh
    /// prompt of another conversation that decodes with speculation), yet C0 drafts over its own conversation, exactly as when
    /// it runs alone (the ring C0 leaves holds the same rows, bit for bit), and every stream is plain decoding's. Before, a lone
    /// ring went to whichever request came next with its rows: C0 drafted over X1's rows at the positions the two shared. On a
    /// device whose room the model makes (<see cref="ManagesItsDevice"/>): X1 gets a ring beside the kept context only there.
    /// </summary>
    [Fact]
    public async Task AnotherConversationInBetweenLeavesATurnItsContext()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        int[][] conversation = Conversation(0);
        int[] other = Conversation(1)[1];
        int[] plain = (await ServeConversation(target, draft, BackendType.Cpu, speculate: false, conversation)).Outputs[2];
        int[] otherPlain = (await ServeConversation(target, draft, BackendType.Cpu, speculate: false, new[] { other, other, other })).Outputs[2];
        var alone = await ServeAndKeep(target, draft, 0, conversation);

        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        ManagesItsDevice(model);
        using var engine = new InferenceEngine(model, Config(speculate: true), NullLogger.Instance);
        var outputs = new Dictionary<string, (int[] Tokens, int Reused)>();
        foreach (var (id, prompt, maxNew) in new[] { ("c0t0", conversation[0], 1), ("c0t1", conversation[1], 1), ("c1t2", other, CMaxNew),
                     ("c0t2", conversation[2], CMaxNew) })
        {
            var served = await Serve(engine, id, prompt, maxNew).WaitAsync(TimeSpan.FromMinutes(3));
            _output.WriteLine($"{id}: reused {served.Completion.PrefixCacheReusedTokens} of {prompt.Length}, " +
                $"{engine.ExecutorForTests.DraftContextsContinued} continued, {model.DFlashRingCountForTests} rings, kept run {model.DFlashRingRunForTests(id)}");
            outputs[id] = (served.Tokens, served.Completion.PrefixCacheReusedTokens);
        }
        Assert.Equal(conversation[2].Length, outputs["c0t2"].Reused);
        Assert.Equal(2, engine.ExecutorForTests.DraftContextsContinued);
        Assert.Equal(otherPlain, outputs["c1t2"].Tokens);
        Assert.Equal(plain, outputs["c0t2"].Tokens);
        Assert.Equal(alone.Tokens, outputs["c0t2"].Tokens);
        float[][] ring = KeptRing(model, "c0t2");
        Assert.Equal(alone.Ring.Length, ring.Length);
        for (int l = 0; l < ring.Length; l++)
            Assert.True(ring[l].AsSpan().SequenceEqual(alone.Ring[l]), $"layer {l / 2}'s ring rows differ from the conversation's run alone");
    }

    /// <summary>
    /// Two conversations' turns interleaved (A0, A1, B0, B1, C1, C0), each turn continuing its conversation with the token
    /// the turn before drew (as a chat's history holds the answer): both keep their contexts in rings of their own, each C
    /// leaves the ring it leaves when its conversation runs alone, bit for bit, and every stream is plain decoding's. On a
    /// device whose room the model makes (<see cref="ManagesItsDevice"/>): the second conversation gets a ring beside the
    /// first's kept context only there.
    /// </summary>
    [Fact]
    public async Task InterleavedConversationsEachKeepTheirContext()
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var (target, draft) = Fixtures();
        var plain = new int[2][];
        var alone = new (int[] Tokens, float[][] Ring)[2];
        for (int c = 0; c < 2; c++)
        {
            plain[c] = (await ServeChat(target, draft, speculate: false, new[] { (c, 0), (c, 1), (c, 2) })).Outputs[(c, 2)];
            var run = await ServeChat(target, draft, speculate: true, new[] { (c, 0), (c, 1), (c, 2) }, managed: true);
            Assert.Equal(2, run.Continued);
            alone[c] = (run.Outputs[(c, 2)], run.Rings[c]);
        }
        var both = await ServeChat(target, draft, speculate: true, new[] { (0, 0), (1, 0), (0, 1), (1, 1), (1, 2), (0, 2) }, managed: true, _output);
        Assert.Equal(4, both.Continued);
        Assert.Equal(2, both.RingCount);
        for (int c = 0; c < 2; c++)
        {
            Assert.Equal(plain[c], both.Outputs[(c, 2)]);
            Assert.Equal(alone[c].Tokens, both.Outputs[(c, 2)]);
            Assert.Equal(alone[c].Ring.Length, both.Rings[c].Length);
            for (int l = 0; l < both.Rings[c].Length; l++)
                Assert.True(both.Rings[c][l].AsSpan().SequenceEqual(alone[c].Ring[l]),
                    $"conversation {c}: layer {l / 2}'s ring rows differ from its run alone");
        }
    }

    /// <summary>Conversations' turns in <paramref name="order"/> on one engine: turn 0 a 40-token prompt and one token, turn 1
    /// that prompt, its token and 20 more, one token, turn 2 turn 1's prompt and its token, <see cref="CMaxNew"/> tokens
    /// (with speculation where on). Each turn's tokens, the contexts continued, the rings, and the rings the conversations'
    /// last turns kept. <paramref name="managed"/>: on a device whose room the model makes (<see cref="ManagesItsDevice"/>).</summary>
    private static async Task<(Dictionary<(int, int), int[]> Outputs, int Continued, int RingCount, Dictionary<int, float[][]> Rings)> ServeChat(
        string target, string draft, bool speculate, (int Conversation, int Turn)[] order, bool managed = false,
        Xunit.Abstractions.ITestOutputHelper output = null)
    {
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        if (managed) ManagesItsDevice(model);
        using var engine = new InferenceEngine(model, Config(speculate), NullLogger.Instance);
        var outputs = new Dictionary<(int, int), int[]>();
        var prompts = new Dictionary<(int, int), int[]>();
        var rings = new Dictionary<int, float[][]>();
        foreach (var (c, t) in order)
        {
            int[] prompt = t == 0 ? Conversation(c)[0]
                : t == 1 ? prompts[(c, 0)].Concat(outputs[(c, 0)]).Concat(Conversation(c)[1][40..]).ToArray()
                : prompts[(c, 1)].Concat(outputs[(c, 1)]).ToArray();
            prompts[(c, t)] = prompt;
            var served = await Serve(engine, $"c{c}t{t}", prompt, t == 2 ? CMaxNew : 1).WaitAsync(TimeSpan.FromMinutes(3));
            outputs[(c, t)] = served.Tokens;
            output?.WriteLine($"c{c}t{t}: reused {served.Completion.PrefixCacheReusedTokens} of {prompt.Length}, " +
                $"{engine.ExecutorForTests.DraftContextsContinued} continued, {model.DFlashRingCountForTests} rings, kept run {model.DFlashRingRunForTests($"c{c}t{t}")}");
            // Each turn reuses its previous turn's end state (a prompt that is all of it forwards its last token again).
            int expected = t == 0 ? 0 : Math.Min(prompts[(c, t - 1)].Length + 1, prompt.Length - 1);
            Assert.True(served.Completion.PrefixCacheReusedTokens == expected,
                $"c{c}t{t} reused {served.Completion.PrefixCacheReusedTokens}, not its previous turn's {expected}");
            if (t == 2 && speculate) rings[c] = KeptRing(model, $"c{c}t{t}");
        }
        return (outputs, engine.ExecutorForTests.DraftContextsContinued, model.DFlashRingCountForTests, rings);
    }

    // ---------------------------------------------------------------- helpers

    private const int CMaxNew = 12;

    private (string Target, string Draft) Fixtures()
    {
        string target = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        string draft = Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "dflash2.gguf"), blockSize: 5,
            hidden: Qwen35DenseSyntheticModelBuilder.Hidden, layers: 2, slidingWindow: 16, dflash2: true);
        return (target, draft);
    }

    /// <summary>A conversation's three prompts: A (40 tokens), B = A and 20 more, C = B.</summary>
    private static int[][] Conversation(int c)
    {
        int[] a = Enumerable.Range(0, 40).Select(i => 40 + (i * 7 + 3 * c) % 23).ToArray();
        int[] b = a.Concat(Enumerable.Range(0, 20).Select(i => 44 + (i * 5 + c) % 19)).ToArray();
        return new[] { a, b, b };
    }

    /// <summary>The prompt in 16-token chunks on the request's own slot, each registered to follow as a request that
    /// speculates or one that does not.</summary>
    private static void Prefill(StandardDecoderModel model, string id, int[] prompt, bool speculates)
    {
        var unified = (IUnifiedBatchModel)model;
        var head = (IDraftHead)model;
        unified.EnsureSequenceSlot(id, prompt.Length + 32);
        for (int start = 0; start < prompt.Length; start += 16)
        {
            int end = Math.Min(prompt.Length, start + 16);
            if (speculates) head.FollowPlainPrompt(id, prompt.Length);
            else head.FollowUnspeculatedPrompt(id, prompt.Length);
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt[start..end] }, new[] { start }, new float[1][]));
        }
    }

    /// <summary>The context rows of positions <paramref name="from"/> .. <paramref name="to"/> of two requests' rings.</summary>
    private static void CompareContexts(StandardDecoderModel a, string aId, StandardDecoderModel b, string bId, string label,
        List<string> mismatches, int from, int to)
    {
        int kvHeads = a.DFlashSettings.NumKVHeads, hd = a.DFlashSettings.HeadDim, row = kvHeads * hd;
        for (int layer = 0; layer < a.DFlashSettings.NumLayers; layer++)
        {
            var (ak, av) = a.DFlashRingContextForTests(aId, layer, out int aFirst);
            var (bk, bv) = b.DFlashRingContextForTests(bId, layer, out int bFirst);
            if (ak == null || bk == null)
            {
                mismatches.Add($"{label}: layer {layer} has no ring");
                continue;
            }
            if (aFirst != from || ak.Length != (to - from) * row)
            {
                mismatches.Add($"{label}: layer {layer} holds {ak.Length / row} rows from {aFirst}, expected {to - from} from {from}");
                continue;
            }
            int offset = (from - bFirst) * row;
            if (offset < 0 || offset + ak.Length > bk.Length
                || !ak.AsSpan().SequenceEqual(bk.AsSpan(offset, ak.Length)) || !av.AsSpan().SequenceEqual(bv.AsSpan(offset, av.Length)))
                mismatches.Add($"{label}: layer {layer}'s rows differ");
        }
    }

    // Eight running sequences: the prefix cache keeps as many end states, enough for two conversations' turns and their
    // prompt-end checkpoints (at four, the first conversation's last state goes before its third turn).
    private static SchedulerConfig Config(bool speculate) => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 16,
        NumBlocks = 64,
        BlockSize = 16,
        EnablePrefixCaching = true,
        StopRepetition = false,
        Speculation = speculate
            ? new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.Block, MaxDraftTokens = 3, MinDraftProb = 0 }
            : new SpeculationOptions { Enabled = false },
    };

    private static async Task<(int[] Tokens, InferenceCompletion Completion, SequenceState Seq)> Serve(InferenceEngine engine, string id,
        int[] prompt, int maxNew)
    {
        // Each conversation its own scope, as the server scopes a client's conversation.
        var seq = new SequenceState(id, prompt, maxNew, 16, SamplingConfig.Greedy, cacheScope: id[..2]);
        var handle = engine.SubmitRequest(seq);
        var tokens = new List<int>();
        await foreach (int t in handle.Tokens.ReadAllAsync())
            tokens.Add(t);
        var completion = await handle.Completion;
        await engine.WhenStepSettledAsync();
        return (tokens.ToArray(), completion, seq);
    }

    private static async Task<(int[][] Outputs, int Continued)> ServeConversation(string target, string draft, BackendType backend,
        bool speculate, int[][] turns)
    {
        using var model = new Qwen35RuntimeModel(target, backend, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        using var engine = new InferenceEngine(model, Config(speculate), NullLogger.Instance);
        var outputs = new int[turns.Length][];
        for (int t = 0; t < turns.Length; t++)
            outputs[t] = (await Serve(engine, $"c0t{t}", turns[t], t == 2 ? CMaxNew : 1).WaitAsync(TimeSpan.FromMinutes(3))).Tokens;
        return (outputs, engine.ExecutorForTests.DraftContextsContinued);
    }

    /// <summary>A conversation served alone, turn by turn, on a device whose room the model makes; C's stream and the ring C
    /// kept.</summary>
    private static async Task<(int[] Tokens, float[][] Ring)> ServeAndKeep(string target, string draft, int c, int[][] turns)
    {
        using var model = new Qwen35RuntimeModel(target, BackendType.Cpu, 1, null, 1, draft);
        model.HostMemorySpareForTests = () => null;   // the machine's other programs do not evict its conversations
        ManagesItsDevice(model);
        using var engine = new InferenceEngine(model, Config(speculate: true), NullLogger.Instance);
        int[] tokens = null;
        for (int t = 0; t < 3; t++)
        {
            var served = await Serve(engine, $"c{c}t{t}", turns[t], t == 2 ? CMaxNew : 1).WaitAsync(TimeSpan.FromMinutes(3));
            if (t == 2) tokens = served.Tokens;
        }
        Assert.Equal(2, engine.ExecutorForTests.DraftContextsContinued);
        return (tokens, KeptRing(model, $"c{c}t2"));
    }

    /// <summary>The model makes its device's room itself, as on one direct-CUDA device (StandardDecoderModel.EnsureDeviceRoom):
    /// a stand-in device of 64 GB, all of it free. Only such a model gives kept rings back when short, so only it makes a
    /// ring to keep a context beside the kept ones (RoomForDFlashRing).</summary>
    private static void ManagesItsDevice(StandardDecoderModel model)
    {
        const long gb = 1L << 30;
        model.DeviceRoomForTests = () => (64 * gb, 64 * gb);
    }

    /// <summary>The context rows (K, then V, per draft layer) of the ring kept for <paramref name="requestId"/>.</summary>
    private static float[][] KeptRing(StandardDecoderModel model, string requestId)
    {
        var rows = new List<float[]>();
        for (int layer = 0; layer < model.DFlashSettings.NumLayers; layer++)
        {
            var (k, v) = model.DFlashRingContextForTests(requestId, layer, out _);
            Assert.NotNull(k);
            rows.Add(k);
            rows.Add(v);
        }
        return rows.ToArray();
    }
}
