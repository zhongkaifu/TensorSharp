// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// D01: the unified execution contract supports a request's whole lifecycle on its own. A model that implements only
// IUnifiedBatchModel (execution) with its ISequenceStateStore (state), and optionally IPrefixCacheModel (reuse), is admitted
// by the host and served by the engine - submission, batched execution, prefix reuse, cancellation, a failed forward and
// release - without implementing IBatchedPagedModel or the KV snapshot contract.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class UnifiedContractStandaloneTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-unified-only-" + Guid.NewGuid().ToString("N"));

    public UnifiedContractStandaloneTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static SchedulerConfig Configuration(bool prefixCache = true) => new()
    {
        BlockSize = 8, NumBlocks = 256, MaxNumBatchedTokens = 64, MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64, EnablePrefixCaching = prefixCache, StopRepetition = false,
    };

    private static List<int> Tokens(int count, int start = 1) => Enumerable.Range(start, count).ToList();

    private static SequenceState Request(string id, List<int> tokens, int maxNew = 8, string? scope = "conversation")
        => new(id, tokens, maxNew, 8, SamplingConfig.Greedy, cacheScope: scope);

    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
    {
        InferenceCompletion completion = await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(20));
        return completion;
    }

    /// <summary>The greedy tokens the oracle's hash chain gives a prompt: what any correct serving of it produces.</summary>
    private static List<int> Expected(IReadOnlyList<int> prompt, int count, int vocab = OracleModel.DefaultVocab)
    {
        var cache = new OracleCache();
        foreach (int t in prompt) cache.Append(t);
        var output = new List<int>();
        for (int i = 0; i < count; i++)
        {
            int next = OracleHash.PeakToken(cache.State, vocab);
            output.Add(next);
            cache.Append(next);
        }
        return output;
    }

    [Fact]
    public void TheContractIsTheUnifiedOne()
    {
        var model = new UnifiedOnlyOracle();
        Assert.False(((IModelArchitecture)model).SupportsKVStateSnapshot);
        Assert.IsNotAssignableFrom<IBatchedPagedModel>(model);
        Assert.Equal(EngineContract.Unified, EngineServability.ContractOf(model));
        Assert.True(EngineServability.CanServe(model));
        Assert.Null(EngineServability.RefusalReason(model));
        // A model with no contract is refused with a reason.
        var none = new UnifiedOnlyOracle(unified: false);
        Assert.Equal(EngineContract.None, EngineServability.ContractOf(none));
        Assert.Contains("no serving contract", EngineServability.RefusalReason(none));
    }

    [Fact]
    public async Task ARequestRunsOnItsStateAndIsReleased()
    {
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration(prefixCache: false));
        Assert.True(engine.ExecutorForTests.ServesUnifiedThroughStateStore);
        var request = Request("one", Tokens(40));
        InferenceCompletion completion = await Run(engine, request);
        Assert.Equal(Expected(request.PromptTokens, 8), request.OutputTokens);
        Assert.True(model.UnifiedForwards > 0);
        Assert.Equal(0, model.PrimaryForwards);   // nothing ran on a primary cache
        Assert.Empty(model.LiveSlots);
        Assert.Contains("one", model.Released);
        Assert.Equal(0, completion.PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task ConcurrentRequestsBatchInOneForwardAndEachIsExact()
    {
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration(prefixCache: false));
        var requests = Enumerable.Range(0, 4).Select(i => Request("r" + i, Tokens(20 + 3 * i, 1 + 17 * i))).ToArray();
        var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
        await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(20));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(20));
        foreach (var r in requests)
            Assert.Equal(Expected(r.PromptTokens, 8), r.OutputTokens);
        Assert.True(model.MaxRequestsInOneForward >= 2, $"the requests never shared a forward ({model.MaxRequestsInOneForward})");
        Assert.Empty(model.LiveSlots);
    }

    [Fact]
    public async Task AFinishedRequestsStateIsReusedExactly()
    {
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration());
        Assert.True(engine.PrefixCacheActive);
        var first = Request("turn-1", Tokens(40));
        await Run(engine, first);
        // The next turn continues the conversation: its prompt is the first turn's tokens and three more.
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens).Concat(new[] { 101, 102, 103 }).ToList();
        var second = Request("turn-2", prompt);
        InferenceCompletion completion = await Run(engine, second);
        Assert.True(completion.PrefixCacheReusedTokens > 0, "the second turn reused nothing");
        Assert.Equal(Expected(prompt, 8), second.OutputTokens);
        Assert.True(model.Donations > 0);
        Assert.True(model.Materializations > 0);
        Assert.Empty(model.LiveSlots);
    }

    [Fact]
    public async Task ACancelledRequestReleasesItsState()
    {
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration(prefixCache: false));
        var gate = new ManualResetEventSlim(false);
        model.BeforeForward = ids => { if (ids.Contains("long") && model.UnifiedForwards >= 3) gate.Wait(TimeSpan.FromSeconds(10)); };
        var request = Request("long", Tokens(24), maxNew: 64);
        var handle = engine.SubmitRequest(request);
        // Abort while the request decodes.
        await Task.Run(() => SpinWait.SpinUntil(() => model.UnifiedForwards >= 3, TimeSpan.FromSeconds(10)));
        engine.Abort("long");
        gate.Set();
        InferenceCompletion completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(20));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(20));
        Assert.True(request.OutputTokens.Count < 64);
        Assert.Empty(model.LiveSlots);
        Assert.Contains("long", model.Released);
        Assert.NotNull(completion);
    }

    [Fact]
    public async Task AFailedForwardFailsItsRequestAndReleasesItsState()
    {
        // A failed forward fails the requests it carried and frees their state; the engine serves the next request.
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration(prefixCache: false));
        model.FailFor = "bad";
        var bad = Request("bad", Tokens(12));
        await Assert.ThrowsAnyAsync<Exception>(() => engine.SubmitRequest(bad).Completion.WaitAsync(TimeSpan.FromSeconds(20)));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(20));
        Assert.NotNull(bad.Error);
        Assert.Contains("bad", model.Released);
        Assert.Empty(model.LiveSlots);

        var good = Request("good", Tokens(18, 50));
        await Run(engine, good);
        Assert.Equal(Expected(good.PromptTokens, 8), good.OutputTokens);
        Assert.Empty(model.LiveSlots);
    }

    [Fact]
    public async Task AStateLostBeforeItsFirstStepIsForwardedAgainOnItsSlot()
    {
        // A store that hands back fewer tokens than the request computed (its reused state was reclaimed between
        // admission and the first step) has the lost tail forwarded again on the request's own slot, never on a primary.
        var model = new UnifiedOnlyOracle();
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("turn-1", Tokens(40));
        await Run(engine, first);
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens).Concat(new[] { 7, 8, 9 }).ToList();
        model.LoseNextMaterializedState = true;
        var second = Request("turn-2", prompt);
        await Run(engine, second);
        Assert.Equal(Expected(prompt, 8), second.OutputTokens);
        Assert.Equal(0, model.PrimaryForwards);
        Assert.Empty(model.LiveSlots);
    }

    [Fact]
    public async Task TheHostServesAModelWithOnlyTheUnifiedContract()
    {
        UnifiedOnlyHostModel loaded = null;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => loaded = new UnifiedOnlyHostModel(path));
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Configuration(false) };
        lifecycle.LoadModel(WriteMinimalGguf("unified.gguf"), null, "cpu");
        Assert.False(loaded.SupportsKVStateSnapshot);
        Assert.IsNotAssignableFrom<IBatchedPagedModel>(loaded);

        InferenceEngine engine = host.TryGetEngine();
        Assert.NotNull(engine);   // before D01 the host refused it (no snapshot, no IBatchedPagedModel)
        Assert.Same(loaded, engine.Model);
        var request = Request("hosted", Tokens(30));
        await Run(engine, request);
        Assert.Equal(Expected(request.PromptTokens, 8), request.OutputTokens);
        Assert.Empty(loaded.Store.LiveSlots);
    }

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_dir, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u);
        bw.Write(3u);
        bw.Write(0UL);
        bw.Write(0UL);
        bw.Write(new byte[8]);
        return path;
    }

    /// <summary>A <see cref="ModelBase"/> that serves only through the unified contract (an <see cref="UnifiedOnlyOracle"/>'s
    /// slots): what a new family would implement to be served by the host.</summary>
    private sealed class UnifiedOnlyHostModel : ModelBase, IUnifiedBatchModel
    {
        internal readonly UnifiedOnlyOracle Store = new();

        public UnifiedOnlyHostModel(string ggufPath) : base(ggufPath, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "unified-only", NumLayers = 1, VocabSize = OracleModel.DefaultVocab };
            _maxContextLength = Store.MaxContextLength;
            Tokenizer = new OracleTokenizer(OracleModel.DefaultVocab);
        }
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException("no primary cache");
        protected override void ResetKVCacheCore() { }

        public bool SupportsUnifiedBatching => true;
        public int EnsureSequenceSlot(string requestId, int reserveTokens) => Store.EnsureSequenceSlot(requestId, reserveTokens);
        public bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
            float[][]? features = null) => Store.TryForwardUnified(requestIds, tokens, starts, lastRowLogits, features);
        public float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start) => Store.ForwardUnifiedMedia(requestId, tokens, start);
        public bool HasSequenceState(string requestId) => Store.HasSequenceState(requestId);
        public bool TryRewindSequenceState(string requestId, int tokens) => Store.TryRewindSequenceState(requestId, tokens);
        public void ReleaseSequenceState(string requestId) => Store.ReleaseSequenceState(requestId);
    }
}

/// <summary>
/// A model that implements only the unified execution contract: <see cref="IUnifiedBatchModel"/> over per-request slots (each
/// an <see cref="OracleCache"/>: a hash chain whose greedy tokens any wrong reuse changes), its <see cref="ISequenceStateStore"/>,
/// and the reuse capability (<see cref="IPrefixCacheModel"/>: a finished slot donated, a donated or copied slot cloned or
/// re-keyed into a new request's slot). No primary cache, no <see cref="IBatchedPagedModel"/>, no KV snapshots.
/// </summary>
internal sealed class UnifiedOnlyOracle : IModelArchitecture, IUnifiedBatchModel, IPrefixCacheModel
{
    private readonly bool _unified;
    private readonly Dictionary<string, OracleCache> _slots = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OracleCache> _retained = new(StringComparer.Ordinal);

    internal UnifiedOnlyOracle(bool unified = true)
    {
        _unified = unified;
        Config = new ModelConfig { VocabSize = OracleModel.DefaultVocab, Architecture = "unified-only-oracle" };
        Tokenizer = new OracleTokenizer(OracleModel.DefaultVocab);
    }

    internal IReadOnlyCollection<string> LiveSlots => _slots.Keys.ToArray();
    internal List<string> Released { get; } = new();
    internal int UnifiedForwards;
    internal int PrimaryForwards;
    internal int MaxRequestsInOneForward;
    internal int Donations;
    internal int Materializations;
    internal string? FailFor;
    internal bool LoseNextMaterializedState;
    internal Action<IReadOnlyList<string>>? BeforeForward;

    // ------------------------------------------------------------------ IModelArchitecture (no primary cache)

    public ModelConfig Config { get; }
    public ITokenizer Tokenizer { get; }
    public IMultimodalInjector MultimodalInjector => null!;
    public IBackendExecutionPlan ExecutionPlan => null!;
    public int MaxContextLength => 4096;
    public string KVStateFingerprint => "unified-only-oracle|v1";
    public bool SupportsKVCacheTruncation => true;

    public float[] Forward(int[] tokens)
    {
        PrimaryForwards++;
        throw new NotSupportedException("A unified-only model has no primary cache.");
    }

    public void ResetKVCache() { }
    public void TruncateKVCache(int tokenCount) => throw new NotSupportedException("A unified-only model has no primary cache.");
    public bool TryTruncateKVCache(int tokenCount) => false;
    public void Dispose() { }

    // ------------------------------------------------------------------ IUnifiedBatchModel + ISequenceStateStore

    public bool SupportsUnifiedBatching => _unified;

    public int EnsureSequenceSlot(string requestId, int reserveTokens)
    {
        if (!_slots.TryGetValue(requestId, out OracleCache? slot))
            _slots[requestId] = slot = new OracleCache();
        return slot.Length;
    }

    public bool HasSequenceState(string requestId) => requestId != null && _slots.ContainsKey(requestId);

    public bool TryRewindSequenceState(string requestId, int tokens)
    {
        if (!_slots.TryGetValue(requestId, out OracleCache? slot) || tokens < 0 || tokens > slot.Length) return false;
        slot.TruncateTo(tokens, corrupt: false);
        return true;
    }

    public void ReleaseSequenceState(string requestId)
    {
        Released.Add(requestId);
        _slots.Remove(requestId);
    }

    public bool TryForwardUnified(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] starts, float[][] lastRowLogits,
        float[][]? features = null)
    {
        BeforeForward?.Invoke(requestIds);
        for (int s = 0; s < requestIds.Count; s++)
            if (!_slots.TryGetValue(requestIds[s], out OracleCache? slot) || slot.Length != starts[s]) return false;
        if (FailFor != null && requestIds.Contains(FailFor))
            throw new InvalidOperationException($"the fixture fails every forward of {FailFor}");
        UnifiedForwards++;
        MaxRequestsInOneForward = Math.Max(MaxRequestsInOneForward, requestIds.Count);
        for (int s = 0; s < requestIds.Count; s++)
        {
            OracleCache slot = _slots[requestIds[s]];
            foreach (int t in tokens[s]) slot.Append(t);
            slot.Flush();
            lastRowLogits[s] = OracleHash.Logits(slot.State, Config.VocabSize);
            if (features != null) features[s] = null!;
        }
        return true;
    }

    public float[] ForwardUnifiedMedia(string requestId, int[] tokens, int start)
    {
        var logits = new float[1][];
        if (!TryForwardUnified(new[] { requestId }, new[] { tokens }, new[] { start }, logits))
            throw new InvalidOperationException($"{requestId}: its slot does not continue at {start}");
        return logits[0];
    }

    // ------------------------------------------------------------------ IPrefixCacheModel (state reuse)

    public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
    {
        Class = FamilyClass.S,
        NamespaceFingerprint = KVStateFingerprint,
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        PrimaryResident = false,
        MinRetainTokens = 8,
        Truncation = TruncationKind.Any,
        RewindCapTokens = int.MaxValue,
        Pages = PageSupport.None,
    };

    public void AttachPrefixCache(IPrefixPayloadSink sink) { }

    public bool CapturesUnboundRequests => true;

    public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        if (!_slots.TryGetValue(requestId, out OracleCache? slot) || _retained.ContainsKey(payloadKey)) return false;
        _retained[payloadKey] = slot.CopyHost();
        footprint = Measure(_retained[payloadKey]);
        return true;
    }

    public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        if (!_slots.TryGetValue(requestId, out OracleCache? slot) || _retained.ContainsKey(payloadKey) || length <= 0 || length > slot.Length)
            return false;
        if (length < slot.Length) slot.TruncateTo(length, corrupt: false);
        _slots.Remove(requestId);
        _retained[payloadKey] = slot;
        Donations++;
        footprint = Measure(slot);
        return true;
    }

    public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    public bool TryMaterialize(in MaterializeRequest request)
    {
        if (!CanMaterialize(request.PayloadKey, request.PayloadTokens, request.TargetTokens)) return false;
        if (string.IsNullOrEmpty(request.TargetRequestId) || _slots.ContainsKey(request.TargetRequestId)) return false;
        OracleCache source = _retained[request.PayloadKey];
        OracleCache target;
        switch (request.Op)
        {
            case MaterializeOp.Clone:
                target = source.CopyHost();
                break;
            case MaterializeOp.Donate:
                _retained.Remove(request.PayloadKey);
                target = source;
                break;
            default:
                return false;
        }
        if (LoseNextMaterializedState)
        {
            // The state was reclaimed after admission: the request's slot comes back empty.
            LoseNextMaterializedState = false;
            target = new OracleCache();
        }
        _slots[request.TargetRequestId] = target;
        Materializations++;
        return true;
    }

    public bool TryReturnDonation(string requestId, string payloadKey)
    {
        if (!_slots.Remove(requestId, out OracleCache? slot) || _retained.ContainsKey(payloadKey)) return false;
        _retained[payloadKey] = slot;
        return true;
    }

    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        => payloadKey != null && _retained.TryGetValue(payloadKey, out OracleCache? cache) && cache.Length == payloadTokens
           && targetTokens >= 0 && targetTokens <= payloadTokens;

    public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
    {
        foreach (string key in payloadKeys)
            if (key != null) _retained.Remove(key);
    }

    public PayloadFootprint MeasureEndState(string payloadKey)
        => payloadKey != null && _retained.TryGetValue(payloadKey, out OracleCache? cache) ? Measure(cache) : default;

    private static PayloadFootprint Measure(OracleCache cache)
        => new(cache.Length, cache.CapacityTokens, new ResourceVector { HostKv = 16L * cache.CapacityTokens }, cache.RopeDelta);

    public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
        => payloadKey != null && _retained.TryGetValue(payloadKey, out OracleCache? cache)
            ? new ResourceVector { HostKv = 16L * cache.CapacityTokens } : default;

    public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
    public bool TryExport(string payloadKey, Stream destination) => false;

    public bool TryImport(string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    public bool TryBeginImport(int tokens, out object? importTicket)
    {
        importTicket = null;
        return false;
    }

    public bool RunImportRead(object importTicket, Stream source) => false;

    public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    public void AbortImport(object importTicket) { }
    public long QuerySpareBytes(ResourceClass cls) => -1;
}
