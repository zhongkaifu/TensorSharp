// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Collections.Concurrent;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Requests being prepared for submission (<see cref="InferenceEngine.BeginPreparation"/>): an engine with nothing running holds
/// the step that would start the requests already submitted until those being prepared were submitted, so requests sent
/// together prefill in one step; a preparation that ends without a submission lets the step go at once, a lone request
/// never waits, and running requests are never held.
/// </summary>
public sealed class EnginePreparationTests
{
    private static readonly TimeSpan Long = TimeSpan.FromSeconds(30);

    [Fact]
    public async Task ARequestStillBeingPreparedPrefillsInTheFirstStepBesideTheOneSubmittedBeforeIt()
    {
        using var model = new SlotModel();
        using var engine = new InferenceEngine(model, Config()) { PreparationWait = Long };
        var steps = new ConcurrentQueue<string>();
        engine.ScheduledForTests = output => steps.Enqueue(Describe(output));

        IDisposable preparing = engine.BeginPreparation();
        var first = engine.SubmitRequest(Sequence("a"));
        // The engine would have started the first prompt by now: it holds it for the request being prepared.
        await Task.Delay(200);
        Assert.Empty(steps);
        var second = engine.SubmitRequest(Sequence("b"));
        preparing.Dispose();

        await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(steps.TryPeek(out string firstStep));
        Assert.Equal("a:P b:P", firstStep);
        Assert.Equal(1, engine.StepsHeldForPreparation);
    }

    [Fact]
    public async Task APreparationEndedWithoutASubmissionReleasesTheHeldStepAtOnce()
    {
        using var model = new SlotModel();
        using var engine = new InferenceEngine(model, Config()) { PreparationWait = Long };
        IDisposable preparing = engine.BeginPreparation();
        var request = engine.SubmitRequest(Sequence("a"));
        await Task.Delay(100);
        Assert.False(request.Completion.IsCompleted);
        var released = System.Diagnostics.Stopwatch.StartNew();
        preparing.Dispose();
        preparing.Dispose();   // a second end changes nothing
        await request.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(released.Elapsed < TimeSpan.FromSeconds(10));
        Assert.Equal(1, engine.StepsHeldForPreparation);
    }

    [Fact]
    public async Task ALoneRequestIsNeverHeld()
    {
        using var model = new SlotModel();
        using var engine = new InferenceEngine(model, Config()) { PreparationWait = Long };
        IDisposable preparing = engine.BeginPreparation();
        preparing.Dispose();   // its own preparation ends when it is submitted
        var completion = await engine.SubmitRequest(Sequence("a")).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(4, completion.OutputTokenCount);
        Assert.Equal(0, engine.StepsHeldForPreparation);
    }

    [Fact]
    public async Task RunningRequestsAreNotHeldForOneBeingPrepared()
    {
        // Each forward takes 20 ms, so the request is still running when the other's preparation begins.
        using var model = new SlotModel { ForwardDelay = TimeSpan.FromMilliseconds(20) };
        using var engine = new InferenceEngine(model, Config()) { PreparationWait = Long };
        var running = engine.SubmitRequest(Sequence("a", output: 16));
        // Running: its first token streamed.
        Assert.True(await running.Tokens.WaitToReadAsync().AsTask().WaitAsync(TimeSpan.FromSeconds(10)));
        using IDisposable preparing = engine.BeginPreparation();
        var completion = await running.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(16, completion.OutputTokenCount);
        Assert.Equal(0, engine.StepsHeldForPreparation);
    }

    private static string Describe(SchedulerOutput output)
        => string.Join(" ", output.ScheduledWork.Select(work => work.Sequence.RequestId + (work.IsPrefill ? ":P" : ":D")));

    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 16,
        NumBlocks = 64,
        BlockSize = 8,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
        StopRepetition = false,
    };

    private static SequenceState Sequence(string id, int prompt = 8, int output = 4)
        => new(id, Enumerable.Repeat(1, prompt).ToList(), output, 8, SamplingConfig.Greedy);

    /// <summary>A model whose requests run on per-request slots, one forward each (no device, no graphs).</summary>
    private sealed class SlotModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly HashSet<string> _slots = new(StringComparer.Ordinal);
        public ModelConfig Config { get; } = new() { Architecture = "qwen35", VocabSize = 16 };
        public ITokenizer Tokenizer { get; } = new StubTokenizer();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => false;
        public bool SupportsCrossSequenceKvReuse => false;
        public int MaxContextLength => 256;
        public bool BatchedForwardAvailable => false;
        public bool SupportsPerSequenceFusedForward => true;
        public bool CanBatchDecode(string requestId, int position) => false;
        public TimeSpan ForwardDelay { get; init; }
        public float[] Forward(int[] tokens)
        {
            if (ForwardDelay > TimeSpan.Zero) Thread.Sleep(ForwardDelay);
            var logits = new float[16];
            logits[7] = 100;
            return logits;
        }
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
            => throw new InvalidOperationException("The slots have no batched graph.");
        public bool BindSequenceCache(string id) => _slots.Add(id);
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
