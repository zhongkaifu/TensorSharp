// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server.ResponseSerializers;

namespace InferenceWeb.Tests;

public sealed class GenerationTimingTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-timing-{Guid.NewGuid():N}.gguf");

    public GenerationTimingTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose() => File.Delete(_path);

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public async Task ChatAndGenerate_ReportActualPrefillAndDecode_IncludingEosOnly(bool generate, bool eosOnly)
    {
        TimingModel model = null;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => model = new TimingModel(path) { EosOnly = eosOnly });
        lifecycle.LoadModel(_path, null, "cpu");
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance)
        {
            SchedulerConfigOverride = Config(),
        };
        var pipeline = new ChatGenerationPipeline(lifecycle, host,
            new KVCachePromptRenderer(new FixedRenderer()),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        using var session = new ChatSession();
        var updates = generate
            ? pipeline.GenerateStreamAsync(session, "question", null, 2, CancellationToken.None, SamplingConfig.Greedy)
            : pipeline.ChatStreamWithMetricsAsync(session,
                new List<ChatMessage> { new() { Role = "user", Content = "question" } },
                2, CancellationToken.None, SamplingConfig.Greedy);
        ChatStreamUpdate terminal = default;
        var text = new StringBuilder();
        await foreach (var update in updates)
        {
            if (update.Done)
                terminal = update;
            else if (update.Piece.Length > 0)
            {
                text.Append(update.Piece);
                // Slow streaming consumers must affect total latency, not the
                // measured model decode throughput.
                await Task.Delay(120);
            }
        }

        Assert.True(terminal.Done);
        Assert.Equal(eosOnly ? "" : "xx", text.ToString());
        Assert.Equal(eosOnly ? 0 : 2, terminal.EvalTokens);
        Assert.Equal(8, terminal.PromptTokens);
        Assert.Equal(4, model.PrefillCalls); // all four chunks must be accumulated
        Assert.InRange(terminal.PromptNs,
            InferenceTelemetry.ToNanos(model.PrefillTicks),
            InferenceTelemetry.ToNanos(model.PrefillTicks) + 50_000_000);
        Assert.InRange(terminal.EvalNs,
            InferenceTelemetry.ToNanos(model.DecodeTicks),
            InferenceTelemetry.ToNanos(model.DecodeTicks) + 50_000_000);
        Assert.True(terminal.PromptNs >= 100_000_000, "Prompt timing omitted the model prefill.");
        Assert.True(terminal.EvalNs > 0, "EOS-only requests still execute a decode forward.");
        Assert.True(terminal.TotalNs >= terminal.PromptNs + terminal.EvalNs);
        if (!eosOnly)
            Assert.True(terminal.TotalNs - terminal.PromptNs - terminal.EvalNs >= 150_000_000,
                "Client backpressure should remain outside model compute durations.");

        using var wire = JsonDocument.Parse(JsonSerializer.Serialize(
            OllamaResponseFactory.GenerateNonStreamingResponse("timing", text.ToString(), terminal.FinishReason,
                terminal.PromptTokens, terminal.EvalTokens, terminal.KvCacheReusedTokens,
                terminal.TotalNs, terminal.PromptNs, terminal.EvalNs)));
        Assert.Equal(terminal.PromptNs, wire.RootElement.GetProperty("prompt_eval_duration").GetInt64());
        Assert.Equal(terminal.EvalNs, wire.RootElement.GetProperty("eval_duration").GetInt64());
        Assert.Equal(terminal.TotalNs, wire.RootElement.GetProperty("total_duration").GetInt64());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SuccessfulFusedBatch_ReportsEachSequencesShareOfCompute(bool sampled)
    {
        using var model = new TimingModel(_path) { SampledBatch = sampled };
        var cfg = Config();
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var scheduler = new ContinuousBatchScheduler(cfg, pool,
            NullLogger.Instance, supportsCrossSequenceKvReuse: false);
        var executor = new BatchExecutor(model, pool, scheduler, NullLogger.Instance);
        var step = new SchedulerOutput();
        foreach (string id in new[] { "a", "b" })
        {
            var seq = new SequenceState(id, new List<int> { 2, 3 }, 1, cfg.BlockSize, SamplingConfig.Greedy);
            foreach (var block in pool.AllocateNew(2)!) seq.BlockTable.AppendBlock(block);
            Assert.True(model.BindSequenceCache(id));
            seq.LastLogits = TimingModel.Logits();
            seq.AdvanceComputedTokens(2);
            seq.Status = SequenceStatus.Running;
            step.ScheduledWork.Add(new ScheduledSequenceWork(seq, 1, isNewAdmission: false, isPrefill: false));
        }

        var results = executor.ExecuteStep(step);

        Assert.Equal(1, model.BatchCalls);
        Assert.Equal(2, results.Count);
        Assert.All(results, result =>
        {
            Assert.Null(result.Error);
            Assert.False(result.IsPrefill);
            Assert.Equal((int)'x', result.SampledToken);
            Assert.True(result.ForwardElapsedTicks >= model.BatchTicks / 2);
        });
        Assert.Equal(results[0].ForwardElapsedTicks, results[1].ForwardElapsedTicks);
        Assert.True(results.Sum(result => result.ForwardElapsedTicks) < model.BatchTicks + Stopwatch.Frequency / 20,
            "The shared batch should be apportioned once, not charged in full to every sequence.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Completion_PreservesPhaseTimes_WhenSpeculationEmitsSeveralTokensOrRequestAborts(bool aborted)
    {
        using var model = new TimingModel(_path);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var seq = new SequenceState("timed-window", new List<int> { 2, 3 }, 8, 2, SamplingConfig.Greedy);
        var handle = new InferenceRequestHandle(seq, engine, CancellationToken.None);
        handle.RecordForwardTime(new SequenceStepResult
        {
            Sequence = seq, IsPrefill = true, TokensForwarded = 2, ForwardElapsedTicks = 123,
        });
        handle.RecordForwardTime(new SequenceStepResult
        {
            Sequence = seq, TokensForwarded = 4, SampledToken = 4,
            ExtraTokens = new[] { 5, 6, 7 }, ForwardElapsedTicks = 456,
        });
        // Cache-recovery replay may run while preparing a declined batch, with
        // no scheduled result for this sequence. Completion still includes it
        // exactly once, in the phase of the replayed tokens.
        seq.ReplayPrefillElapsedTicks = 1234;
        seq.ReplayDecodeElapsedTicks = 5678;
        if (aborted) handle.CompleteAborted(); else handle.CompleteFinished();

        var completion = await handle.Completion;
        Assert.Equal(1357, completion.PrefillElapsedTicks);
        Assert.Equal(6134, completion.DecodeElapsedTicks);
    }

    private static SchedulerConfig Config() => new()
    {
        BlockSize = 2, NumBlocks = 128, MaxNumRunningSequences = 2,
        MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 2, SoloPrefillChunkSize = 2,
        EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
    };

    private sealed class FixedRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
            => "abcdefgh";
    }

    private sealed class TimingModel : ModelBase, IBatchedPagedModel
    {
        private readonly HashSet<string> _bound = new();
        public bool EosOnly { get; init; }
        public bool SampledBatch { get; init; }
        public long PrefillTicks, DecodeTicks, BatchTicks;
        public int PrefillCalls, BatchCalls;

        public TimingModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new TimingTokenizer();
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "generation-timing";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        public bool SupportsPerSequenceFusedForward => true;
        public bool BindSequenceCache(string id) => _bound.Add(id);
        public bool HasFusedSequenceCache(string id) => _bound.Contains(id);
        public bool CanBatchDecode(string id, int position) => true;
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context) => throw new NotSupportedException();

        protected override float[] ForwardCore(int[] tokens)
        {
            long started = Stopwatch.GetTimestamp();
            Thread.Sleep(tokens.Length > 1 ? 30 : 10);
            long elapsed = Stopwatch.GetTimestamp() - started;
            if (tokens.Length > 1) { PrefillTicks += elapsed; PrefillCalls++; }
            else DecodeTicks += elapsed;
            return Logits(EosOnly ? 1 : 'x');
        }

        public bool TryForwardBatchedFusedDecodeSampled(
            IReadOnlyList<string> ids, int[] tokens, int[] positions, int[] output)
        {
            if (!SampledBatch) return false;
            MeasureBatch();
            Array.Fill(output, (int)'x');
            return true;
        }

        public bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> ids, int[] tokens, int[] positions, float[][] output)
        {
            MeasureBatch();
            for (int i = 0; i < output.Length; i++) output[i] = Logits();
            return true;
        }

        private void MeasureBatch()
        {
            long started = Stopwatch.GetTimestamp();
            Thread.Sleep(40);
            BatchTicks = Stopwatch.GetTimestamp() - started;
            BatchCalls++;
        }

        internal static float[] Logits(int peak = 'x')
        {
            var result = new float[128];
            result[peak] = 10;
            return result;
        }

        protected override void ResetKVCacheCore() { }
    }

    private sealed class TimingTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
