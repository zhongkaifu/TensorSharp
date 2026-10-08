// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class ChatPipelineReadVisibilityTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ActualTextOrMediaCompactionSignalsBeforeSubmittingGeneration(bool media)
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-compaction-" + Guid.NewGuid().ToString("N") + ".gguf");
        BackendFailureWarmupTests.WriteProbeGguf(path);
        try
        {
            ProbeModel model = null!;
            using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
                (file, backend, tp, draft) => model = new ProbeModel(file));
            lifecycle.LoadModel(path, null, "cpu");
            using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance)
            {
                SchedulerConfigOverride = new SchedulerConfig
                {
                    BlockSize = 8, NumBlocks = 32, MaxNumRunningSequences = 1,
                    MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 16,
                    EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
                },
            };
            using var pipeline = new ChatGenerationPipeline(lifecycle, host, new KVCachePromptRenderer(new Renderer()),
                new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
            using var session = new ChatSession();
            var history = new List<ChatMessage>
            {
                new() { Role = "system", Content = "policy" },
                new() { Role = "user", Content = "repair", ImagePaths = media ? new List<string> { "synthetic-image.png" } : null },
                new() { Role = "assistant", Content = new string('a', media ? 70 : 200) },
                new() { Role = "tool", Content = new string('b', media ? 70 : 200) },
                new() { Role = "assistant", Content = "reread" },
                new() { Role = "tool", Content = "bad anchor" },
            };
            int forwards = model.Forwards;
            await using var stream = pipeline.ChatStreamWithMetricsAsync(session, history, 8, CancellationToken.None)
                .GetAsyncEnumerator();
            Assert.True(await stream.MoveNextAsync());
            Assert.True(stream.Current.HistoryCompacted);
            Assert.False(stream.Current.Done);
            Assert.Empty(stream.Current.Piece);
            Assert.Equal(forwards, model.Forwards);
            Assert.Equal(media, model.Expansions > 0);
            if (media) Assert.True(model.Expansions >= 2); // Initial expansion then compacted media preparation.
        }
        finally { File.Delete(path); }
    }

    private sealed class Renderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false) =>
            string.Join("\n", messages.Select(message => message.Content)) + "\n";
    }

    private sealed class ProbeModel : ModelBase, IVisionCapableModel, IMultimodalPromptExpander
    {
        public int Forwards { get; private set; }
        public int Expansions { get; private set; }
        public ProbeModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "probe", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new CharacterTokenizer();
            _maxContextLength = 256;
        }
        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;
        protected override float[] ForwardCore(int[] tokens)
        {
            Forwards++;
            var logits = new float[128]; logits['X'] = 10; return logits;
        }
        protected override void ResetKVCacheCore() { }
        public bool IsVisionEncoderLoaded => true;
        public void LoadVisionEncoder(string path) => throw new NotSupportedException();
        public void SetVisionEmbeddings(TensorSharp.Tensor embeddings, int position) => throw new NotSupportedException();
        public List<int> ExpandMultimodalPrompt(ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
        {
            Expansions++;
            return tokens.Concat(Enumerable.Repeat((int)'i', 200)).ToList();
        }
    }

    private sealed class CharacterTokenizer : ITokenizer
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
