// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>ignore_eos</c> (OpenAI-compatible, as vLLM, SGLang and llama.cpp spell it): a fixed-length
/// benchmark request runs to its token limit through end-of-sequence draws. The EOS tokens are
/// forwarded and counted like any other token, and their marker text is not answer text.
/// </summary>
public sealed class IgnoreEosTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-ignore-eos-{Guid.NewGuid():N}.gguf");

    public IgnoreEosTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);
    public void Dispose() => File.Delete(_path);

    [Theory]
    [InlineData("""{"ignore_eos": true}""", true)]
    [InlineData("""{"ignore_eos": false}""", false)]
    [InlineData("""{"ignore_eos": null}""", false)]
    [InlineData("""{}""", false)]
    public void ParseOpenAI_ReadsIgnoreEos(string json, bool expected)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.True(SamplingConfigParser.TryValidateIgnoreEos(doc.RootElement, out string error), error);
        SamplingConfig parsed = SamplingConfigParser.ParseOpenAI(doc.RootElement, new SamplingConfig());
        Assert.Equal(expected, parsed.IgnoreEos);
        // Only the length cap ends such a request, so the repetition guard steps aside too.
        Assert.Equal(!expected, parsed.StopRepetition);
        Assert.Equal(expected, parsed.Clone().IgnoreEos);
    }

    [Theory]
    [InlineData("""{"ignore_eos": "true"}""")]
    [InlineData("""{"ignore_eos": 1}""")]
    public void ANonBooleanIgnoreEos_IsRefused(string json)
    {
        using var doc = JsonDocument.Parse(json);
        Assert.False(SamplingConfigParser.TryValidateIgnoreEos(doc.RootElement, out string error));
        Assert.Equal("ignore_eos must be a boolean.", error);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task TheEngine_RunsToTheLimitThroughEosDraws(bool ignoreEos)
    {
        using var model = new AlternatingEosModel(_path);
        using var engine = new InferenceEngine(model, Config());
        SamplingConfig sampling = SamplingConfig.Greedy;
        sampling.IgnoreEos = ignoreEos;
        var seq = new SequenceState("r", "abcdefgh".Select(c => (int)c).ToList(), 6, 2, sampling);

        var handle = engine.SubmitRequest(seq);
        var published = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync())
            published.Add(token);
        InferenceCompletion completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(20));

        if (ignoreEos)
        {
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
            Assert.Equal(6, completion.OutputTokenCount);
            Assert.Equal(new[] { 'a', 1, 'a', 1, 'a', 1 }, published.ToArray());
        }
        else
        {
            Assert.Equal(SequenceStatus.FinishedStopped, completion.Status);
            Assert.Equal("eos", completion.FinishReason);
            Assert.Equal(new int[] { 'a' }, published.ToArray());
        }
    }

    [Fact]
    public async Task ThePipeline_CountsEosTokensButSendsNoMarkerText()
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => new AlternatingEosModel(path));
        lifecycle.LoadModel(_path, null, "cpu");
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance) { SchedulerConfigOverride = Config() };
        var pipeline = new ChatGenerationPipeline(lifecycle, host,
            new KVCachePromptRenderer(new FixedRenderer()),
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
        using var session = new ChatSession();
        SamplingConfig sampling = SamplingConfig.Greedy;
        sampling.IgnoreEos = true;
        var history = new List<ChatMessage> { new() { Role = "user", Content = "go" } };

        var streamed = new StringBuilder();
        ChatStreamUpdate terminal = default;
        await foreach (var update in pipeline.ChatStreamWithMetricsAsync(session, history, 6, CancellationToken.None, sampling))
        {
            if (update.Done) terminal = update;
            else streamed.Append(update.Piece);
        }

        Assert.Equal("aaa", streamed.ToString());
        Assert.Equal(6, terminal.EvalTokens);
        Assert.Equal("max_tokens", terminal.FinishReason);
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

    /// <summary>Answers 'a', then EOS (token 1), then 'a' again: an EOS after every word.</summary>
    private sealed class AlternatingEosModel : ModelBase
    {
        public AlternatingEosModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new EosTokenizer();
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "ignore-eos";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

        protected override float[] ForwardCore(int[] tokens)
        {
            var logits = new float[128];
            logits[tokens[^1] == 'a' ? 1 : 'a'] = 10;
            return logits;
        }

        protected override void ResetKVCacheCore() { }
    }

    private sealed class EosTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => i == 1 ? "<|im_end|>" : ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => string.Concat(tokens.Select(id => Vocab[id]));
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[token]));
    }
}
