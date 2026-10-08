// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class ModelTokenSuppressionTests
{
    [Fact]
    public void ModelSamplingEntryPointsRespectExclusionsAndConfiguredGreedyPenalties()
    {
        string path = Path.Combine(Path.GetTempPath(), $"model-suppression-{Guid.NewGuid():N}.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        try
        {
            using var model = new SamplingModel(path);
            Assert.Equal(1, model.SampleGreedy([100, 5, 4]));
            Assert.Throws<InvalidOperationException>(() => model.SampleGreedy(
                [100, float.NegativeInfinity, float.NegativeInfinity]));
            var config = SamplingConfig.Greedy;
            config.RepetitionPenalty = 2;
            Assert.Equal(2, model.Sample([100, 5, 4], config, new[] { 1 }));
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData("gemma4")]
    [InlineData("llama")]
    public void MetadataExclusionsReachBothTokenizerFamiliesWithoutExcludingAllControlTokens(string family)
    {
        using var fixture = new VocabularyFixture(family);
        fixture.Gguf.Metadata["tokenizer.ggml.suppress_tokens"] = new[] { 1, 1, -1, 99 };
        var tokenizer = ModelBase.CreateTokenizerFromGguf(fixture.Gguf);
        Assert.Equal(new[] { 1 }, tokenizer.SuppressedTokenIds);
        Assert.Equal(2, new TokenSampler(SamplingConfig.Greedy, tokenizer.SuppressedTokenIds).Sample([0, 100, 5, 1]));
        // The legal tool control token and EOG remain possible output.
        Assert.Equal(3, new TokenSampler(SamplingConfig.Greedy, tokenizer.SuppressedTokenIds).Sample([0, 100, 1, 5]));
    }

    [Theory]
    [InlineData("gemma4")]
    [InlineData("llama")]
    public void MissingMetadataDoesNotInventAnArchitectureWideSpecialTokenBan(string family)
    {
        using var fixture = new VocabularyFixture(family);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(fixture.Gguf);
        Assert.Empty(tokenizer.SuppressedTokenIds);
        Assert.Equal(1, new TokenSampler(SamplingConfig.Greedy, tokenizer.SuppressedTokenIds).Sample([0, 100, 5, 1]));
    }

    [Fact]
    public void MalformedSuppressionMetadataFailsInsteadOfSilentlyIgnoringTheContract()
    {
        using var fixture = new VocabularyFixture("gemma4");
        fixture.Gguf.Metadata["tokenizer.ggml.suppress_tokens"] = "1";
        Assert.Throws<InvalidDataException>(() => ModelBase.CreateTokenizerFromGguf(fixture.Gguf));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    public void ExclusionPrecedesGreedyPenaltiesAndProbabilityFilters(float temperature)
    {
        for (int seed = 0; seed < 32; seed++)
        {
            var config = new SamplingConfig { Temperature = temperature, TopK = 1, TopP = .1f,
                MinP = .5f, RepetitionPenalty = 2, PresencePenalty = 1, FrequencyPenalty = 1, Seed = seed };
            var sampler = new TokenSampler(config, new[] { 0 });
            Assert.False(sampler.IsPlainGreedyArgmax);
            Assert.Equal(1, sampler.Sample([100, 5, -10], new[] { 1, 1 }));
        }
    }

    [Fact]
    public void FirstTokenConstraintAndForcedThinkingCannotReintroduceSuppressedTokens()
    {
        var config = SamplingConfig.Greedy;
        config.FirstTokenAllowList = new[] { 0, 1 };
        Assert.Equal(1, new TokenSampler(config, new[] { 0 }).Sample([100, 5]));
        config.FirstTokenAllowList = new[] { 0 };
        Assert.Throws<InvalidOperationException>(() => new TokenSampler(config, new[] { 0 }).Sample([100, 5]));
        config.FirstTokenAllowList = null;
        config.ThinkingBudget = new ThinkingTokenBudget(1, 0);
        Assert.Throws<InvalidOperationException>(() => new TokenSampler(config, new[] { 0 }).Sample([100, 5], new[] { 1 }));
        Assert.Throws<InvalidOperationException>(() => new TokenSampler(SamplingConfig.Greedy, new[] { 0, 1 }).Sample([100, 5]));
    }

    [Theory]
    [InlineData(0f, 0)]
    [InlineData(0f, 1)]
    [InlineData(1f, 0)]
    [InlineData(1f, 1)]
    public void PenaltyOverflowFailsClosedAndRestoresLegalLogits(float temperature, int suppressedId)
    {
        int legalId = 1 - suppressedId;
        var config = new SamplingConfig { Temperature = temperature, TopK = 1, RepetitionPenalty = 2 };
        var sampler = new TokenSampler(config, new[] { suppressedId });
        float[] logits = new float[2];
        logits[suppressedId] = 100;
        logits[legalId] = -float.MaxValue;
        Assert.Throws<InvalidOperationException>(() => sampler.Sample(logits, new[] { legalId }));
        Assert.Equal(-float.MaxValue, logits[legalId]);
        Assert.Equal(float.NegativeInfinity, logits[suppressedId]);

        // A failed draw does not poison reusable score/penalty buffers.
        logits[legalId] = 5;
        Assert.Equal(legalId, sampler.Sample(logits, new[] { legalId }));
    }

    [Fact]
    public void TemperatureOverflowCannotReturnAForbiddenFallbackToken()
    {
        var config = new SamplingConfig { Temperature = float.Epsilon, TopK = 0, TopP = 1 };
        var sampler = new TokenSampler(config, new[] { 0 });
        Assert.Throws<InvalidOperationException>(() => sampler.Sample([100, -1]));
    }

    [Fact]
    public void BindingModelContractInvalidatesPreviouslyCachedDeviceArgmaxAndSampler()
    {
        var sequence = new SequenceState("test", new[] { 0 }, 3, 4, SamplingConfig.Greedy);
        var before = sequence.GetOrCreateSampler();
        Assert.True(before.IsPlainGreedyArgmax);
        sequence.PendingDeviceToken = 1;
        sequence.BindGenerationVocabulary(new BpeTokenizer(["a", "<audio|>"], [1, 3], [], -1, [], false, false)
            { SuppressedTokenIds = new[] { 1 } });
        Assert.Null(sequence.PendingDeviceToken);
        Assert.NotSame(before, sequence.GetOrCreateSampler());
        Assert.False(sequence.GetOrCreateSampler().IsPlainGreedyArgmax);
        float[] draft = [5, 100];
        sequence.GetOrCreateSampler().ApplyModelSuppression(draft);
        Assert.Equal(float.NegativeInfinity, draft[1]);
    }

    [Fact]
    public void EmptyOrEquivalentContractPreservesPendingDeviceTokenAndSamplerState()
    {
        var sequence = new SequenceState("stable-contract", new[] { 0 }, 3, 4,
            new SamplingConfig { Temperature = 1, Seed = 17 });
        var original = sequence.GetOrCreateSampler();
        sequence.PendingDeviceToken = 2;
        sequence.PendingDevicePosition = 4;
        var tokenizer = new BpeTokenizer(["a", "b", "c"], [1, 1, 1], [], -1, [], false, false);
        sequence.BindGenerationVocabulary(tokenizer);
        sequence.BindGenerationVocabulary(null);
        Assert.Same(original, sequence.GetOrCreateSampler());
        Assert.Equal(2, sequence.PendingDeviceToken);
        Assert.Equal(4, sequence.PendingDevicePosition);

        sequence.BindGenerationVocabulary(new BpeTokenizer(["a", "b", "c"], [1, 1, 1], [], -1, [], false, false)
            { SuppressedTokenIds = new[] { 1 } });
        Assert.Null(sequence.PendingDeviceToken);
        var restricted = sequence.GetOrCreateSampler();
        Assert.NotSame(original, restricted);
        sequence.PendingDeviceToken = 2;
        sequence.BindGenerationVocabulary(new BpeTokenizer(["a", "b", "c"], [1, 1, 1], [], -1, [], false, false)
            { SuppressedTokenIds = new[] { 1 } });
        Assert.Same(restricted, sequence.GetOrCreateSampler());
        Assert.Equal(2, sequence.PendingDeviceToken);

        // Removing exclusions changes the distribution and still invalidates
        // the old pending draw; only semantically unchanged bindings are inert.
        sequence.BindGenerationVocabulary(tokenizer);
        Assert.Null(sequence.PendingDeviceToken);
        Assert.NotSame(restricted, sequence.GetOrCreateSampler());
    }

    [Fact]
    public async Task EngineUsesModelContractOnEveryStepWithoutRepetitionPenalties()
    {
        using var model = new SuppressionModel();
        using var engine = new InferenceEngine(model, new SchedulerConfig { BlockSize = 4, NumBlocks = 32,
            EnablePrefixCaching = false, StopRepetition = false, Speculation = SpeculationOptions.Disabled });
        var sequence = new SequenceState("suppression", new[] { 0 }, 5, 4, SamplingConfig.Greedy);
        var handle = engine.SubmitRequest(sequence);
        var output = new List<int>();
        using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token)) output.Add(token);
        await handle.Completion.WaitAsync(deadline.Token);
        Assert.Equal(new[] { 2 }, output);
        Assert.Null(sequence.Error);
    }

    private sealed class SuppressionModel : IModelArchitecture
    {
        private int _position;
        public ModelConfig Config { get; } = new() { Architecture = "gemma4", VocabSize = 4 };
        public ITokenizer Tokenizer { get; } = new BpeTokenizer(["a", "<audio|>", "<tool_call>", "</s>"], [1, 3, 3, 3], [], -1, [3], false, false)
            { SuppressedTokenIds = new[] { 1 } };
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => false;
        public string KVStateFingerprint => "suppression-test";
        public float[] Forward(int[] tokens) { _position += tokens.Length; return _position == 1 ? [0, 100, 5, 1] : [0, 100, 1, 5]; }
        public void ResetKVCache() => _position = 0;
        public void TruncateKVCache(int count) => throw new NotSupportedException();
        public long ComputeKVBlockByteSize(int count) => 0;
        public bool TryExtractKVBlock(int start, int count, Span<byte> destination) => false;
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => false;
        public void Dispose() { }
    }

    private sealed class SamplingModel : ModelBase
    {
        public SamplingModel(string path) : base(path, BackendType.Cpu)
            => Tokenizer = new BpeTokenizer(["<audio|>", "a", "b"], [3, 1, 1], [], -1, [], false, false)
                { SuppressedTokenIds = new[] { 0 } };
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }

    private sealed class VocabularyFixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), $"token-suppression-{Guid.NewGuid():N}.gguf");
        public GgufFile Gguf { get; }
        public VocabularyFixture(string family)
        {
            using (var stream = File.Create(_path))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            { writer.Write(0x46554747u); writer.Write(3u); writer.Write(0UL); writer.Write(0UL); writer.Write(0UL); }
            Gguf = new GgufFile(_path);
            Gguf.Metadata["general.architecture"] = "gemma4";
            Gguf.Metadata["tokenizer.ggml.model"] = family;
            Gguf.Metadata["tokenizer.ggml.tokens"] = new[] { "a", "<audio|>", "<tool_call>", "</s>" };
            Gguf.Metadata["tokenizer.ggml.token_type"] = new[] { 1, 3, 3, 3 };
            Gguf.Metadata["tokenizer.ggml.scores"] = new float[4];
            Gguf.Metadata["tokenizer.ggml.merges"] = Array.Empty<string>();
            Gguf.Metadata["tokenizer.ggml.bos_token_id"] = 0u;
            Gguf.Metadata["tokenizer.ggml.eos_token_id"] = 3u;
        }
        public void Dispose() { Gguf.Dispose(); File.Delete(_path); }
    }
}
