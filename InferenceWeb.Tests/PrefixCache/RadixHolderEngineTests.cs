using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

public class RadixHolderEngineTests
{
    private static SchedulerConfig Configuration(bool enabled = true) => new()
    {
        BlockSize = 8, NumBlocks = 256, MaxNumBatchedTokens = 64,
        MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = enabled, StopRepetition = false,
    };

    private static List<int> Tokens(int count, int start = 1) => Enumerable.Range(start, count).ToList();

    private static SequenceState Request(string id, List<int> tokens, string? scope = "conversation",
        int shared = 0, IReadOnlyList<int>? breaks = null) => new(id, tokens, 3, 8, SamplingConfig.Greedy,
            cacheScope: scope, sharedPrefixTokens: shared, cacheBreakpoints: breaks);

    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
        => await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FinishedHolder_ReusesExactState_WithSequentialRequestIdReuse(bool recurrent)
    {
        var model = recurrent ? OracleFakes.R(8) : OracleFakes.S(8);
        using var engine = new InferenceEngine(model, Configuration());
        Assert.Equal(PrefixCacheMode.Tree, engine.PrefixCacheMode);
        var first = Request("repeat", Tokens(32));
        await Run(engine, first);
        var prompt = first.PromptTokens.Concat(first.OutputTokens)
            .Take(first.NumComputedTokens).Concat(new[] { 101, 102, 103 }).ToList();
        var second = Request("repeat", prompt);
        var completion = await Run(engine, second);
        Assert.Equal(first.NumComputedTokens, completion.PrefixCacheReusedTokens);

        using var cold = new InferenceEngine(recurrent ? OracleFakes.R(8) : OracleFakes.S(8), Configuration(false));
        var baseline = Request("cold", prompt);
        await Run(cold, baseline);
        Assert.Equal(baseline.OutputTokens, second.OutputTokens);
        Assert.All(model.RetainedPayloadKeys, key => Assert.StartsWith("pc:", key));
    }

    [Fact]
    public async Task PublicCheckpoint_IsClonedAcrossScopes_AndSurvivesEngineRestart()
    {
        var store = new MemoryCheckpointStore();
        using (var engine = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            engine.PrefixCheckpointStore = store;
            await Run(engine, Request("seed", Tokens(32), "a", shared: 16));
            var prompt = Tokens(16).Concat(Tokens(16, 81)).ToList();
            var second = Request("other", prompt, "b", shared: 16);
            Assert.Equal(16, (await Run(engine, second)).PrefixCacheReusedTokens);
        }
        Assert.NotNull(store.Bytes);
        using var restarted = new InferenceEngine(OracleFakes.R(8), Configuration());
        restarted.PrefixCheckpointStore = store;
        var after = Request("restart", Tokens(16).Concat(Tokens(16, 61)).ToList(), "c", shared: 16);
        Assert.Equal(16, (await Run(restarted, after)).PrefixCacheReusedTokens);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(13)]
    public async Task WholeSystemPromptWarmup_IsPublicAcrossNewSessions_AndPersists(int prefixLength)
    {
        var store = new MemoryCheckpointStore();
        using (var engine = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            engine.PrefixCheckpointStore = store;
            var warmup = new SequenceState("startup-warmup", Tokens(prefixLength), 1, 8,
                SamplingConfig.Greedy, sharedPrefixTokens: prefixLength);
            Assert.Equal(prefixLength, warmup.SharedPrefixTokens);
            await Run(engine, warmup);

            foreach (string scope in new[] { "first-session", "new-session" })
            {
                var prompt = Tokens(prefixLength).Concat(new[] { 81, 82, 83 }).ToList();
                var request = Request(scope, prompt, scope, shared: prefixLength);
                Assert.Equal(prefixLength, (await Run(engine, request)).PrefixCacheReusedTokens);

                using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
                var reference = Request("cold", prompt);
                await Run(cold, reference);
                Assert.Equal(reference.OutputTokens, request.OutputTokens);
            }
        }
        Assert.NotNull(store.Bytes);
        using var restarted = new InferenceEngine(OracleFakes.R(8), Configuration());
        restarted.PrefixCheckpointStore = store;
        var afterRestart = Request("restarted-session", Tokens(prefixLength).Concat(new[] { 91, 92 }).ToList(),
            "restarted-session", shared: prefixLength);
        Assert.Equal(prefixLength, (await Run(restarted, afterRestart)).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task ExplicitNone_DoesNotReadOrPublishPayloads()
    {
        var model = OracleFakes.R(8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Tokens(32), breaks: Array.Empty<int>());
        await Run(engine, first);
        Assert.Empty(model.RetainedPayloadKeys);
        var second = Request("second", Tokens(32), breaks: Array.Empty<int>());
        Assert.Equal(0, (await Run(engine, second)).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task OtherScope_CannotUsePrivateEndState()
    {
        using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
        var first = Request("first", Tokens(32), "a");
        await Run(engine, first);
        var next = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 91, 92 }).ToList();
        Assert.Equal(0, (await Run(engine, Request("other", next, "b"))).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task PrimaryOnlyFamily_ContinuesItsRadixPrefix()
    {
        var traits = new OracleTraits
        {
            Name = "primary", Class = FamilyClass.S, EndState = EndStateSupport.None,
            Pages = PageSupport.None, Truncation = TruncationKind.Any, PrimaryResident = true,
        };
        using var engine = new InferenceEngine(new OracleModel(traits, 8), Configuration());
        var first = Request("first", Tokens(32));
        await Run(engine, first);
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 71, 72 }).ToList();
        var second = Request("second", prompt);
        Assert.Equal(first.NumComputedTokens, (await Run(engine, second)).PrefixCacheReusedTokens);
        using var baseline = new InferenceEngine(new OracleModel(traits, 8), Configuration(false));
        var cold = Request("cold", prompt);
        await Run(baseline, cold);
        Assert.Equal(cold.OutputTokens, second.OutputTokens);
    }

    [Fact]
    public async Task Shutdown_ReleasesAllTreeOwnedPayloads()
    {
        var model = OracleFakes.R(8);
        var engine = new InferenceEngine(model, Configuration());
        await Run(engine, Request("seed", Tokens(32), shared: 16));
        engine.Dispose();
        Assert.Empty(model.RetainedPayloadKeys);
    }

    [Theory]
    [InlineData("TS_RETAINED_FUSED_CACHE", "0", "TS_PREFIX_CHECKPOINTS", "0")]
    [InlineData("TS_RETAINED_FUSED_CACHE_MAX", "0", "TS_PREFIX_CHECKPOINTS_MAX", "0")]
    public async Task CacheOverrides_DisableOwnedPayloadCopies(string privateSetting, string privateValue,
        string publicSetting, string publicValue)
    {
        string? previousPrivate = Environment.GetEnvironmentVariable(privateSetting);
        string? previousPublic = Environment.GetEnvironmentVariable(publicSetting);
        try
        {
            Environment.SetEnvironmentVariable(privateSetting, privateValue);
            Environment.SetEnvironmentVariable(publicSetting, publicValue);
            var model = OracleFakes.R(8);
            using var engine = new InferenceEngine(model, Configuration());
            await Run(engine, Request("seed", Tokens(32), shared: 16));
            var other = Request("other", Tokens(32), "different", shared: 16);
            Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
            Assert.Empty(model.RetainedPayloadKeys);
        }
        finally
        {
            Environment.SetEnvironmentVariable(privateSetting, previousPrivate);
            Environment.SetEnvironmentVariable(publicSetting, previousPublic);
        }
    }

    [Fact]
    public async Task MemoryPressure_EvictsOldConversation_AndRecomputesCorrectly()
    {
        var model = OracleFakes.R(8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Tokens(32), "a");
        await Run(engine, first);
        await Run(engine, Request("latest", Tokens(32, 61), "b"));
        engine.TrimIdleMemory();
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 101, 102 }).ToList();
        var after = Request("after-trim", prompt, "a");
        Assert.Equal(0, (await Run(engine, after)).PrefixCacheReusedTokens);
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
        var reference = Request("reference", prompt);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, after.OutputTokens);
    }

    [Fact]
    public void PublicCheckpoint_CannotRemainAboveHardResourceCap()
    {
        using var model = OracleFakes.R(8);
        model.SpareBytes = 1; // The recurrent state alone needs 1,024 bytes.
        var pool = new BlockPool(32, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool, model.KVStateFingerprint);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var sequence = Request("prefix", Tokens(16), shared: 16);
        foreach (var block in pool.AllocateNew(2)!) sequence.BlockTable.AppendBlock(block);
        model.Forward(sequence.PromptTokens.ToArray());
        sequence.SetComputedTokensForPrefixAdoption(16);
        cache.CaptureCheckpoint(sequence);
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(0, cache.Tree.PublicEndStateCount);
        Assert.True(cache.Tree.Cached.StateSnapshot <= cache.Tree.EffectiveCap(ResourceClass.StateSnapshot));
        cache.Reset();
        pool.Free(sequence.BlockTable.Clear());
    }

    [Fact]
    public void CompletedColdConversations_DoNotAccumulateScopeMetadata()
    {
        using var model = OracleFakes.R(8);
        var pool = new BlockPool(32, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool, model.KVStateFingerprint);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        for (int i = 0; i < 500; i++)
        {
            var request = Request($"request-{i}", Tokens(16), $"conversation-{i}", breaks: Array.Empty<int>());
            cache.EnsureRequest(request);
            cache.ReleaseRequest(request);
        }
        Assert.Equal(0, cache.RequestCount);
        Assert.Equal(1, cache.Tree.Scopes.LiveCount); // Only the public scope remains.
        Assert.Equal(2, cache.Tree.Scopes.Capacity); // The same private slot was reused.
        cache.Reset();
    }

    [Theory]
    [InlineData("TS_SCHED_DISABLE_BATCHED", "1")]
    [InlineData("TS_PER_SEQ_FUSED", "0")]
    public async Task DisablingHolderExecutionRoute_RecomputesWithoutAdoptingUnreadableHolders(string name, string value)
    {
        string? previous = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, value);
            using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
            var first = Request("first", Tokens(32));
            await Run(engine, first);
            var prompt = Tokens(32).Concat(new[] { 81, 82 }).ToList();
            var after = Request("after", prompt);
            await Run(engine, after);
            using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
            var reference = Request("reference", prompt);
            await Run(cold, reference);
            Assert.Equal(reference.OutputTokens, after.OutputTokens);
        }
        finally { Environment.SetEnvironmentVariable(name, previous); }
    }

    [Fact]
    public async Task ShutdownWithActiveHolders_ReleasesRequestsAndCompletesConsumers()
    {
        var gate = new ComputeGate();
        gate.Close();
        var model = new PausingOracle(gate);
        var engine = new InferenceEngine(model, Configuration()) { ComputeGate = gate };
        var first = engine.SubmitRequest(new SequenceState("a", Tokens(32), 100, 8,
            SamplingConfig.Greedy, cacheScope: "a"));
        var second = engine.SubmitRequest(new SequenceState("b", Tokens(32, 61), 100, 8,
            SamplingConfig.Greedy, cacheScope: "b"));
        gate.Open();
        try
        {
            Assert.True(SpinWait.SpinUntil(() => model.ForwardCalls >= 2 && engine.StepsHeldByGate > 0,
                TimeSpan.FromSeconds(5)), "Both requests should pause with live per-request holders.");
            Assert.True(model.PrivateHolderCount > 0);
        }
        finally { engine.Dispose(); }
        Assert.Equal(0, model.PrivateHolderCount);
        Assert.Empty(model.RetainedPayloadKeys);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => first.Completion);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => second.Completion);
    }

    private sealed class PausingOracle(ComputeGate gate) : OracleModel(new OracleTraits
    {
        Name = "paused", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true, Truncation = TruncationKind.None,
    }, 8), IModelArchitecture
    {
        private int _forwardCalls;
        internal int ForwardCalls => Volatile.Read(ref _forwardCalls);
        float[] IModelArchitecture.Forward(int[] tokens)
        {
            var logits = base.Forward(tokens);
            Interlocked.Increment(ref _forwardCalls);
            gate.Close();
            return logits;
        }
    }

    [Fact]
    public async Task AbortingNonBatchedModel_ReleasesRadixRequestMetadata()
    {
        var inner = new OracleModel(new OracleTraits
        {
            Name = "non-batched", Class = FamilyClass.S, Truncation = TruncationKind.Any,
            EndState = EndStateSupport.None, Pages = PageSupport.None,
        }, 8);
        using var model = new NonBatchedOracle(inner);
        using var engine = new InferenceEngine(model, Configuration());
        model.AfterForward = () => engine.Abort("cancelled");
        var request = new SequenceState("cancelled", Tokens(32), 100, 8,
            SamplingConfig.Greedy, cacheScope: "scope");
        var completion = await Run(engine, request);
        Assert.Equal(SequenceStatus.FinishedAborted, completion.Status);
        Assert.Equal(0, Assert.IsType<PrefixCacheCoordinator>(inner.Sink).RequestCount);
    }

    private sealed class NonBatchedOracle(OracleModel inner) : IModelArchitecture, IPrefixCacheModel
    {
        internal Action? AfterForward;
        public ModelConfig Config => inner.Config;
        public ITokenizer Tokenizer => inner.Tokenizer;
        public IMultimodalInjector MultimodalInjector => inner.MultimodalInjector;
        public IBackendExecutionPlan ExecutionPlan => inner.ExecutionPlan;
        public bool SupportsKVCacheTruncation => true;
        public float[] Forward(int[] tokens)
        {
            float[] logits = inner.Forward(tokens);
            Action? callback = AfterForward;
            AfterForward = null;
            callback?.Invoke();
            return logits;
        }
        public void ResetKVCache() => inner.ResetKVCache();
        public void TruncateKVCache(int tokenCount) => inner.TruncateKVCache(tokenCount);
        public void Dispose() => inner.Dispose();
        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => inner.GetPrefixCacheCapabilities();
        public void AttachPrefixCache(IPrefixPayloadSink sink) => inner.AttachPrefixCache(sink);
        public bool TryCaptureCopy(string id, string key, out PayloadFootprint footprint) => inner.TryCaptureCopy(id, key, out footprint);
        public bool TryCaptureDonate(string id, string key, int length, out PayloadFootprint footprint) => inner.TryCaptureDonate(id, key, length, out footprint);
        public bool TryConvertPrimary(string key, int length, out PayloadFootprint footprint) => inner.TryConvertPrimary(key, length, out footprint);
        public bool TryMaterialize(in MaterializeRequest request) => inner.TryMaterialize(request);
        public bool TryReturnDonation(string id, string key) => inner.TryReturnDonation(id, key);
        public bool CanMaterialize(string key, int count, int target) => inner.CanMaterialize(key, count, target);
        public void ReleasePayloads(ReadOnlySpan<string> keys, ReleaseReason reason) => inner.ReleasePayloads(keys, reason);
        public PayloadFootprint MeasureEndState(string key) => inner.MeasureEndState(key);
        public ResourceVector EstimateCloneBytes(string key, int count) => inner.EstimateCloneBytes(key, count);
        public bool TryCopyPagedToHolder(ReadOnlySpan<int> ids, int tokens, string id) => inner.TryCopyPagedToHolder(ids, tokens, id);
        public bool TryExport(string key, Stream stream) => inner.TryExport(key, stream);
        public bool TryImport(string key, int tokens, Stream stream, out PayloadFootprint footprint) => inner.TryImport(key, tokens, stream, out footprint);
        public bool TryBeginImport(int tokens, out object? ticket) => inner.TryBeginImport(tokens, out ticket);
        public bool RunImportRead(object ticket, Stream stream) => inner.RunImportRead(ticket, stream);
        public bool TryCommitImport(object ticket, string key, out PayloadFootprint footprint) => inner.TryCommitImport(ticket, key, out footprint);
        public void AbortImport(object ticket) => inner.AbortImport(ticket);
        public long QuerySpareBytes(ResourceClass cls) => inner.QuerySpareBytes(cls);
    }

    private sealed class MemoryCheckpointStore : IPrefixCheckpointStore
    {
        internal byte[]? Bytes;
        private int[]? _tokens;
        public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out Stream payload)
        {
            if (Bytes is not null && prefixTokens.SequenceEqual(_tokens))
            {
                payload = new MemoryStream(Bytes, writable: false);
                return true;
            }
            payload = null!;
            return false;
        }
        public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<Stream> writePayload)
        {
            using var stream = new MemoryStream();
            writePayload(stream);
            _tokens = prefixTokens.ToArray();
            Bytes = stream.ToArray();
            return true;
        }
    }
}
