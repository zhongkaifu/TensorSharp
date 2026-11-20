// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #383, problem 2: issue #376's long-turn reproducer on the CUDA executor. A is a prompt with one token to generate, B is
// A's transcript (A's reply rendered back verbatim) plus a user turn, again with one token, and C is B's prompt again. C has to
// resume B's whole prompt, never fall back to A's checkpoint and prefill B's turn again (969 tokens at 6.6K-69K on the
// reporter's machine, with a copy of A's checkpoint as device memory was fullest): B's turn was shorter than
// SchedulerConfig.PromptTailCheckpointMinTokens and its reply is rendered back verbatim, which skipped B's prompt-end
// checkpoint; where that checkpoint shares the request's rows (a fixed set of recurrent states) it is always taken.
// Each C answers as a fresh run of its prompt does. The managed CPU executor runs the CUDA executor's slot-bound policies
// through the model's seam (SlotBoundPoliciesForTests), with the share invariants checked before every step.
using Microsoft.Extensions.Logging.Abstractions;

using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class PromptEndCheckpointRetentionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-prompt-end-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;
    private string? _fixture;

    public PromptEndCheckpointRetentionTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private StandardDecoderModel Create(bool slotBound)
    {
        _fixture ??= new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var model = (StandardDecoderModel)ModelBase.Create(_fixture, BackendType.Cpu);
        model.SlotBoundPoliciesForTests = slotBound;
        model.SlotInvariantChecksForTests = true;
        model.RetainCopyMaxBytesForTests = 0;   // long slots: retained by donation, as the 27B's past a few thousand tokens
        return model;
    }

    // The prompt-tail rule scaled to these prompts: the reporter's A (6,603 tokens) protects more than the default 1,024 and
    // B's turn (968) fewer; here A's 300 tokens and B's 61 fall either side of 100.
    private const int TailCheckpointMinTokens = 100;

    private static InferenceEngine Engine(StandardDecoderModel model, bool prefixCache = true) => new(model, new SchedulerConfig
    {
        MaxNumBatchedTokens = 512,
        MaxNumRunningSequences = 4,
        NumBlocks = 256,
        BlockSize = 16,
        EnablePrefixCaching = prefixCache,
        StopRepetition = false,
        PromptTailCheckpointMinTokens = TailCheckpointMinTokens,
    }, NullLogger.Instance);

    private static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 3 + (i * 37 + i / 7 + salt * 11) % 250).ToArray();

    // A no-reasoning chat turn: its reply is rendered back verbatim (ChatGenerationPipeline sets this without a thinking budget).
    private static SequenceState Turn(string id, int[] prompt, int max, bool replyRenderedBack) =>
        new(id, prompt, max, 16, SamplingConfig.Greedy, cacheScope: "conversation") { PromptTailCheckpointOptional = replyRenderedBack };

    private static async Task<int[]> Serve(InferenceEngine engine, SequenceState sequence)
    {
        var handle = engine.SubmitRequest(sequence);
        var tokens = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync()) tokens.Add(token);
        await handle.Completion.WaitAsync(TimeSpan.FromMinutes(2));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromMinutes(1));
        return tokens.ToArray();
    }

    /// <summary>What a fresh engine without a prefix cache answers for <paramref name="prompt"/>.</summary>
    private async Task<int[]> Cold(int[] prompt, int max)
    {
        using StandardDecoderModel model = Create(slotBound: true);
        using InferenceEngine engine = Engine(model, prefixCache: false);
        return await Serve(engine, Turn("cold", prompt, max, replyRenderedBack: true));
    }

    /// <summary>
    /// The reproducer as the reporter ran it: no reasoning, so every reply is rendered back verbatim, and B's turn (61 tokens
    /// here, 968 there) is shorter than the tokens a prompt-tail checkpoint must protect where it copies the request's state
    /// (<see cref="TailCheckpointMinTokens"/> here, 1,024 there), while A's prompt is longer. Where the checkpoint shares B's rows, B takes it and C resumes B's whole prompt; where it would copy them (the
    /// policies without slot-bound recordings), the rule still skips it and C resumes A's checkpoint.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task AShortTurnWhoseReplyIsRenderedBackLeavesItsPromptEndForARetry(bool slotBound)
    {
        int[] a = Tokens(300, 1);
        using StandardDecoderModel model = Create(slotBound);
        int[] b, c, answer;
        int reusedB, reusedC;
        using (InferenceEngine engine = Engine(model))
        {
            int[] first = await Serve(engine, Turn("A", a, 1, replyRenderedBack: true));
            b = a.Concat(first).Concat(Tokens(60, 2)).ToArray();
            var turnB = Turn("B", b, 1, replyRenderedBack: true);
            await Serve(engine, turnB);
            reusedB = turnB.PrefixCacheReusedTokens;
            c = b;
            var turnC = Turn("C", c, 8, replyRenderedBack: true);
            answer = await Serve(engine, turnC);
            reusedC = turnC.PrefixCacheReusedTokens;
        }
        _output.WriteLine($"slotBound={slotBound}: B reused {reusedB}/{b.Length}, C reused {reusedC}/{c.Length}; shared checkpoints " +
            $"{model.SharedCheckpoints}, steals {model.RowSteals}, copies {model.SlotCopies}");
        Assert.Equal(a.Length + 1, reusedB);   // A's end state: its prompt and its one reply token
        if (slotBound)
        {
            Assert.Equal(c.Length, reusedC);
            Assert.Equal(0, model.SharedCheckpointsInvalidatedByWrite);
            model.CheckSlotShareInvariants();
        }
        else
            Assert.Equal(a.Length, reusedC);   // A's prompt-end checkpoint: B's 61 tokens are prefilled again
        Assert.Equal(await Cold(c, 8), answer);
    }
}
