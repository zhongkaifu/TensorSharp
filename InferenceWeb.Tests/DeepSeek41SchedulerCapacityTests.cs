// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]

public sealed class DeepSeek41SchedulerCapacityTests : IDisposable
{
    private readonly EnvScope _env = new();

    public DeepSeek41SchedulerCapacityTests()
    {
        _env.Set("TS_SCHED_NUM_BLOCKS", null);
        _env.ClearSpeculationVars();
    }

    public void Dispose() => _env.Dispose();

    [Theory]
    [InlineData("deepseek41", true, false, false, 132)]
    [InlineData("DEEPSEEK41", true, false, false, 132)]
    [InlineData("deepseek4", true, false, false, 33)]
    [InlineData("qwen35", true, false, false, 33)]
    [InlineData("deepseek41", false, false, false, 33)]
    [InlineData("deepseek41", true, true, false, 33)]
    // A model that declares native sequence slots (the direct-CUDA Qwen 3.8
    // Flash Next engine), whatever its architecture; the declaration alone does
    // not count without the per-sequence path, or with byte snapshots.
    [InlineData("qwen4exp", true, false, true, 132)]
    [InlineData("qwen4exp", true, false, false, 33)]
    [InlineData("qwen4exp", false, false, true, 33)]
    [InlineData("qwen4exp", true, true, true, 33)]
    public void AutoCapacityMatchesNativeSlotOwnershipOnly(
        string architecture, bool perSequence, bool snapshots, bool declaresNativeSlots, int expectedBlocks)
    {
        // Each 257-token slot needs 33 blocks. Rounding the aggregate instead
        // would leave three slots unable to reserve their final partial block.
        using var model = new SlotModel(architecture, 257, perSequence, snapshots, declaresNativeSlots);
        using var engine = new InferenceEngine(model, Config());
        Assert.Equal(expectedBlocks, engine.PoolStats.totalBlocks);
    }

    [Fact]
    public void ExplicitBlockLimitRemainsAnOperatorLimit()
    {
        _env.Set("TS_SCHED_NUM_BLOCKS", "7");
        using var model = new SlotModel();
        var cfg = Config(blocks: 7);
        using var engine = new InferenceEngine(model, cfg);
        Assert.Equal(7, engine.PoolStats.totalBlocks);
    }

    [Fact]
    public void ImpossibleMetadataCapacityFailsBeforeAllocating()
    {
        using var model = new SlotModel(context: int.MaxValue);
        var cfg = Config(slots: int.MaxValue);
        var error = Assert.Throws<InvalidOperationException>(() => new InferenceEngine(model, cfg));
        Assert.Contains("too many KV blocks", error.Message);
    }

    [Fact]
    public async Task FourLongNativeSlotsPrefillExactlyOnceWithoutPoolPreemption()
    {
        using var model = new SlotModel();
        var gate = new ComputeGate();
        gate.Close();
        using var engine = new InferenceEngine(model, Config()) { ComputeGate = gate };
        var sequences = Enumerable.Range(0, 4).Select(i => Sequence($"long-{i}", 200, 4)).ToArray();
        var handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
        gate.Open();

        var completions = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(completions, c => Assert.Equal(4, c.OutputTokenCount));
        Assert.Equal(800, model.PromptTokensForwarded);
        Assert.Equal(4, model.MaxConcurrentSlots);
        Assert.Equal(128, engine.PoolStats.totalBlocks);
        Assert.Equal(128, engine.PoolStats.freeBlocks);
    }

    [Fact]
    public async Task FourLongDeclaredNativeSlotsPrefillExactlyOnceWithoutPoolPreemption()
    {
        // Four 200-token prompts on 256-token slots: a pool of one context (32
        // blocks) would hold one of them and preempt the rest, and a delta
        // recurrence cannot rewind, so each preemption re-prefills from zero.
        using var model = new SlotModel("qwen4exp", declaresNativeSlots: true);
        var gate = new ComputeGate();
        gate.Close();
        using var engine = new InferenceEngine(model, Config()) { ComputeGate = gate };
        var handles = Enumerable.Range(0, 4).Select(i => engine.SubmitRequest(Sequence($"q4e-{i}", 200, 4))).ToArray();
        gate.Open();

        var completions = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(completions, c => Assert.Equal(4, c.OutputTokenCount));
        Assert.Equal(800, model.PromptTokensForwarded);
        Assert.Equal(4, model.MaxConcurrentSlots);
        Assert.Equal(128, engine.PoolStats.totalBlocks);
    }

    [Fact]
    public async Task AggregateCapacityDoesNotEnlargeAnIndividualDeclaredNativeSlot()
    {
        using var model = new SlotModel("qwen4exp", declaresNativeSlots: true);
        using var engine = new InferenceEngine(model, Config());
        var oversized = Sequence("oversized", 253, 4);
        var error = await Assert.ThrowsAsync<ContextLengthExceededException>(() =>
            engine.SubmitRequest(oversized).Completion.WaitAsync(TimeSpan.FromSeconds(10)));

        Assert.Contains("each native sequence slot", error.Message);
        Assert.Equal(256, error.ContextTokens);
        Assert.Equal(0, model.PromptTokensForwarded);
    }

    [Fact]
    public async Task AggregateCapacityDoesNotEnlargeAnIndividualNativeSlot()
    {
        using var model = new SlotModel();
        using var engine = new InferenceEngine(model, Config());
        var oversized = Sequence("oversized", 253, 4);
        // The refusal hosts report in one line (the CLI's configuration error, the server's 400).
        var error = await Assert.ThrowsAsync<ContextLengthExceededException>(() =>
            engine.SubmitRequest(oversized).Completion.WaitAsync(TimeSpan.FromSeconds(10)));

        Assert.Contains("each native sequence slot", error.Message);
        Assert.Equal(256, error.ContextTokens);
        Assert.Equal(SequenceStatus.FinishedError, oversized.Status);
        Assert.Equal(0, oversized.BlockTable.NumBlocks);
        Assert.Equal(0, model.PromptTokensForwarded);
        var valid = await engine.SubmitRequest(Sequence("valid", 252, 4)).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(4, valid.OutputTokenCount);
        Assert.Equal(252, model.PromptTokensForwarded);
    }

    [Theory]
    [InlineData(2048, 512)]
    [InlineData(4096, 1024)]
    public void FourPrefillsShareTheExplicitBatchTokenBudget(int batchTokens, int expectedChunk)
    {
        var cfg = ProfileConfig(batchTokens, 256);
        var pool = new BlockPool(4096, cfg.BlockSize, 0);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        for (int i = 0; i < 4; ++i) scheduler.Submit(Sequence($"prefill-{i}", 8192, 4));
        var step = scheduler.Schedule();
        Assert.Equal(4, step.ScheduledWork.Count);
        Assert.All(step.ScheduledWork, work => Assert.Equal(expectedChunk, work.NumScheduledTokens));
    }

    [Theory]
    [InlineData(256)]
    [InlineData(1024)]
    public void MixedDecodeUsesTheExplicitPrefillCapEvenWithLargerBatchBudget(int mixedChunk)
    {
        var cfg = ProfileConfig(4096, mixedChunk);
        var pool = new BlockPool(4096, cfg.BlockSize, 0);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var decoder = Sequence("decoder", 4, 16);
        scheduler.Submit(decoder);
        var first = Assert.Single(scheduler.Schedule().ScheduledWork);
        decoder.AdvanceComputedTokens(first.NumScheduledTokens);
        for (int i = 0; i < 3; ++i) scheduler.Submit(Sequence($"prefill-{i}", 8192, 4));
        var step = scheduler.Schedule();
        Assert.Equal(4, step.ScheduledWork.Count);
        Assert.Equal(1, step.ScheduledWork[0].NumScheduledTokens);
        Assert.All(step.ScheduledWork.Skip(1), work => Assert.Equal(mixedChunk, work.NumScheduledTokens));
    }

    private static SchedulerConfig Config(int blocks = 4, int slots = 4,
        int batchTokens = 64, int mixedChunk = 16, int soloChunk = 64) => new()
    {
        MaxNumBatchedTokens = batchTokens,
        MaxNumRunningSequences = slots,
        MaxPrefillChunkSize = mixedChunk,
        SoloPrefillChunkSize = soloChunk,
        NumBlocks = blocks,
        BlockSize = 8,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
        StopRepetition = false,
    };

    private static SchedulerConfig ProfileConfig(int batchTokens, int mixedChunk)
        => Config(batchTokens: batchTokens, mixedChunk: mixedChunk, soloChunk: 1024);

    private static SequenceState Sequence(string id, int prompt, int output)
        => new(id, Enumerable.Repeat(1, prompt).ToList(), output, 8, SamplingConfig.Greedy);

    // Reproduces V4.1's independent native holders and metadata-only block pool
    // (or a model declaring the same, as the direct-CUDA Qwen 3.8 Flash Next
    // engine does); no batched graph or byte snapshot is available to mask
    // recomputation.
    private sealed class SlotModel(
        string architecture = "deepseek41", int context = 256,
        bool perSequence = true, bool snapshots = false,
        bool declaresNativeSlots = false) : IModelArchitecture, IBatchedPagedModel
    {
        private readonly HashSet<string> _slots = new(StringComparer.Ordinal);
        public int PromptTokensForwarded { get; private set; }
        public int MaxConcurrentSlots { get; private set; }
        public ModelConfig Config { get; } = new() { Architecture = architecture, VocabSize = 16 };
        public ITokenizer Tokenizer { get; } = new StubTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => snapshots;
        public bool SupportsCrossSequenceKvReuse => false;
        public int MaxContextLength => context;
        public bool BatchedForwardAvailable => false;
        public bool SupportsPerSequenceFusedForward => perSequence;
        public bool UsesNativeSequenceSlots => declaresNativeSlots;
        public bool CanBatchDecode(string requestId, int position) => false;
        public float[] Forward(int[] tokens)
        {
            PromptTokensForwarded += tokens.Count(t => t == 1);
            var logits = new float[16];
            logits[7] = 100;
            return logits;
        }
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
            => throw new InvalidOperationException("A native V4.1 slot has no batched graph.");
        public bool BindSequenceCache(string id)
        {
            bool fresh = _slots.Add(id);
            MaxConcurrentSlots = Math.Max(MaxConcurrentSlots, _slots.Count);
            return fresh;
        }
        public void AdoptPrimaryCacheToFused(string id) => BindSequenceCache(id);
        public bool HasFusedSequenceCache(string id) => _slots.Contains(id);
        public void OnSequenceReleased(string id) => _slots.Remove(id);
        public void RestorePrimaryCache() { }
        public void ResetKVCache() { }
        public void TruncateKVCache(int count) { }
        public void Dispose() { }
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, 16).Select(i => i.ToString()).ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => [];
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int id, List<byte> buffer) => buffer.AddRange(System.Text.Encoding.UTF8.GetBytes(id.ToString()));
        public bool IsEos(int id) => false;
        public int LookupToken(string text) => -1;
    }
}
