// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// A turn the chat pipeline stops itself (a stop sequence, the thinking budget) is aborted in the
/// engine, which may already have forwarded tokens past the last one streamed. Those tokens are in the
/// cache, so the transcript must record them or the next turn's render diverges from the cache at the end
/// of this answer. Client cancellations were handled; the pipeline's own stops were not, and a model that
/// cannot rewind (Qwen 3.5/3.6/3.8) then reused only the previous prompt, or nothing: 39 of 502 and 0 of
/// 3052 tokens on Qwen3.5-9B after a thinking-budget stop, 490 of 503 once recorded.
/// </summary>
public sealed class ChatPipelineSelfStopTranscriptTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-selfstop-{Guid.NewGuid():N}.gguf");

    public ChatPipelineSelfStopTranscriptTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose() => File.Delete(_path);

    [Fact]
    public async Task AStopSequenceStop_RecordsEveryTokenTheEngineForwarded()
    {
        DigitModel model = null;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => model = new DigitModel(path));
        lifecycle.LoadModel(_path, null, "cpu");
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config() };
        var pipeline = new ChatGenerationPipeline(lifecycle, host,
            new KVCachePromptRenderer(new FixedRenderer()),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        using var session = new ChatSession();
        SamplingConfig sampling = SamplingConfig.Greedy;
        sampling.StopSequences = new List<string> { "34" };
        var history = new List<ChatMessage> { new() { Role = "user", Content = "count" } };

        var streamed = new StringBuilder();
        ChatStreamUpdate terminal = default;
        await foreach (var update in pipeline.ChatStreamWithMetricsAsync(session, history, 64, CancellationToken.None, sampling))
        {
            if (update.Done)
            {
                terminal = update;
            }
            else if (update.Piece.Length > 0)
            {
                streamed.Append(update.Piece);
                await Task.Delay(40);   // a slow client: the engine keeps forwarding ahead of the stream
            }
        }

        Assert.Equal("stop_sequence", terminal.FinishReason);
        Assert.Equal("01234", streamed.ToString());
        int[] forwarded = model.ForwardedAnswerTokens();
        Assert.True(forwarded.Length > streamed.Length,
            $"the engine forwarded {forwarded.Length} tokens, no more than the {streamed.Length} streamed: nothing to prove");

        var next = session.Transcripts.Augment(new List<ChatMessage>
        {
            history[0],
            new() { Role = "assistant", Content = streamed.ToString() },
            new() { Role = "user", Content = "go on" },
        });
        IReadOnlyList<int> recorded = next.History[1].RawOutputTokens;
        Assert.NotNull(recorded);
        // The cache must be a prefix of what the next turn renders for this answer.
        Assert.True(recorded.Count >= forwarded.Length,
            $"recorded {recorded.Count} tokens, the cache holds {forwarded.Length}");
        Assert.Equal(forwarded, recorded.Take(forwarded.Length).ToArray());
    }

    /// <summary>
    /// A turn's raw tokens are filed under the model load that produced them, and only that
    /// load gets them back. After a switch the app keeps the same chat open, and the old
    /// model's ids used to be spliced into the new model's prompt.
    /// </summary>
    [Fact]
    public async Task ATurnIsRecordedUnderTheLoadThatProducedIt()
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => new DigitModel(path));
        lifecycle.LoadModel(_path, null, "cpu");
        lifecycle.LoadModel(_path, null, "cpu");   // a switch: this load's epoch is not the first's
        Assert.NotEqual(0, lifecycle.LoadEpoch);
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config() };
        var pipeline = new ChatGenerationPipeline(lifecycle, host,
            new KVCachePromptRenderer(new FixedRenderer()),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        using var session = new ChatSession();
        var history = new List<ChatMessage> { new() { Role = "user", Content = "count" } };

        var streamed = new StringBuilder();
        await foreach (var update in pipeline.ChatStreamWithMetricsAsync(session, history, 4, CancellationToken.None, SamplingConfig.Greedy))
            if (!update.Done) streamed.Append(update.Piece);

        var followUp = new List<ChatMessage>
        {
            history[0],
            new() { Role = "assistant", Content = streamed.ToString() },
            new() { Role = "user", Content = "go on" },
        };
        Assert.NotNull(session.Transcripts.Augment(followUp, lifecycle.LoadEpoch).History[1].RawOutputTokens);
        Assert.Null(session.Transcripts.Augment(followUp, lifecycle.LoadEpoch + 1).History[1].RawOutputTokens);
    }

    /// <summary>
    /// The model is unloaded while a request is still preparing. A request the switch
    /// cancelled ends as cancelled; one nobody cancelled is told the model went away,
    /// instead of a cancellation a server would read as the client leaving.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AnUnloadDuringPreparationEndsTheRequestCleanly(bool cancelledBySwitch)
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => new DigitModel(path));
        lifecycle.LoadModel(_path, null, "cpu");
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config() };
        using var stop = new CancellationTokenSource();
        bool unloaded = false;
        var renderer = new CallbackRenderer(() =>
        {
            if (unloaded) return;
            unloaded = true;
            if (cancelledBySwitch) stop.Cancel();
            host.Reset();
            lifecycle.Unload();
        });
        var pipeline = new ChatGenerationPipeline(lifecycle, host,
            new KVCachePromptRenderer(renderer),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        using var session = new ChatSession();
        var history = new List<ChatMessage> { new() { Role = "user", Content = "count" } };

        async Task Run()
        {
            await foreach (var _ in pipeline.ChatStreamWithMetricsAsync(session, history, 4, stop.Token, SamplingConfig.Greedy)) { }
        }

        Exception ex = await Record.ExceptionAsync(Run);
        Assert.True(unloaded);
        if (cancelledBySwitch)
            Assert.IsAssignableFrom<OperationCanceledException>(ex);
        else
            Assert.IsType<ModelUnloadedException>(ex);
    }

    private sealed class CallbackRenderer : IPromptRenderer
    {
        private readonly Action _onRender;
        public CallbackRenderer(Action onRender) => _onRender = onRender;

        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
        {
            _onRender();
            return "abcdefgh";
        }
    }

    private static SchedulerConfig Config() => new()
    {
        BlockSize = 2, NumBlocks = 128, MaxNumRunningSequences = 2,
        MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 16,
        EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
    };

    private sealed class FixedRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
            => "abcdefgh";
    }

    /// <summary>Counts 0, 1, 2, ... after the prompt and records every answer token it is fed.</summary>
    private sealed class DigitModel : ModelBase
    {
        private readonly List<int> _forwarded = new();

        public DigitModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new CharTokenizer();
        }

        public int[] ForwardedAnswerTokens()
        {
            lock (_forwarded) return _forwarded.ToArray();
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "self-stop";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_forwarded)
                foreach (int t in tokens)
                    if (t >= '0' && t <= '9' || t == 'z') _forwarded.Add(t);
            int last = tokens[^1];
            var logits = new float[128];
            logits[last >= '0' && last < '9' ? last + 1 : last == '9' || last == 'z' ? 'z' : '0'] = 10;
            return logits;
        }

        protected override void ResetKVCacheCore()
        {
            lock (_forwarded) _forwarded.Clear();
        }
    }

    private sealed class CharTokenizer : ITokenizer
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
