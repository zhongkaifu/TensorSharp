// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 speculating on a BOUND per-request holder - the TensorAgent path,
// where every turn after a chat's first continues a retained holder or a
// shared-prefix checkpoint clone. The speculative trunk used to flip the model's
// per-sequence fused capability off after its first step, the planner re-routed
// the holder-resident sequence to the linear path and the request lost its
// position. Here the same two-turn conversation runs plainly and with n-gram
// speculation through ONE engine (the app's run-time switch), on the real model,
// greedily: the streams must be identical, the second run must have verified
// windows and accepted drafts, and turn two must have run from a reused prefix.
//
//   TS_TEST_MODEL_DIR=~/work/models/Qwen TS_TEST_GGML_BACKEND=metal
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]

public class Qwen35HolderSpeculationTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;
    public Qwen35HolderSpeculationTests(ITestOutputHelper output) { _output = output; }

    private const string Passage =
        "The lighthouse keeper counted the ships each evening, writing their names in a ledger " +
        "bound in green cloth. Seventeen on Monday, twelve on Tuesday, none at all on the night of " +
        "the storm, when the lamp itself seemed to flinch at every gust. By Friday the ledger had " +
        "a new column, for the birds that rested on the gallery rail before crossing the bay. ";

    [ModelFact(EnvModelDir, "qwen3.5-9b-iq4_xs|qwen3.5-9b-q8_0")]
    public async Task NgramSpeculationOnARetainedHolder_GivesThePlainStream()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = dir == null ? null : TestGates.FindGguf(dir, "qwen3.5-9b-iq4_xs|qwen3.5-9b-q8_0");
        if (modelPath == null) { _output.WriteLine("no qwen3.5-9b model; skipping"); return; }
        BackendType backend = TestGates.PinnedGgmlBackend;

        string prevRetained = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX");
        string prevPerSeq = Environment.GetEnvironmentVariable("TS_PER_SEQ_FUSED");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", null);
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        try
        {
            using var model = ModelBase.Create(modelPath, backend);
            Assert.True(((ISpeculativeTarget)model).SpecTrunkFollowsBoundCache, "Qwen 3.5's trunk must follow the bound holder");

            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 1024,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 512,
                SoloPrefillChunkSize = 512,
                NumBlocks = 64,
                BlockSize = 64,
                EnablePrefixCaching = true,
                Speculation = new SpeculationOptions { Enabled = false },
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

            int[] turn1 = model.Tokenizer.Encode(Passage + Passage + "\n\nSummary of the passage in one sentence:").ToArray();
            int[] followUp = model.Tokenizer.Encode("\n\nNow repeat the passage above word for word:\n\n").ToArray();

            // Round one is a CONCURRENT pair: that is what puts each conversation in
            // its own per-request fused holder (a solo request lives in the linear
            // live cache); the follow-up then continues the retained holder, which
            // is the TensorAgent shape (its chats start from a checkpoint clone).
            int[] other = model.Tokenizer.Encode("List five rivers of Europe, one per line:").ToArray();
            async Task<(List<int> t1, List<int> t2, SequenceState seq1, SequenceState seq2, InferenceCompletion c2)> ConversationAsync(string tag)
            {
                // The follow-up continues its own conversation's retained holder: the
                // Radix cache reuses a finished turn only within its conversation scope.
                var s1 = new SequenceState($"{tag}-1", turn1.ToList(), 24, cfg.BlockSize, SamplingConfig.Greedy,
                    cacheScope: tag);
                var sOther = new SequenceState($"{tag}-other", other.ToList(), 24, cfg.BlockSize, SamplingConfig.Greedy,
                    cacheScope: tag + "-other");
                var h1 = engine.SubmitRequest(s1);
                var hOther = engine.SubmitRequest(sOther);
                var r1 = DrainAsync(h1);
                var rOther = DrainAsync(hOther);
                await Task.WhenAll(r1, rOther);
                var (_, out1) = await r1;
                var prompt2 = turn1.Concat(out1).Concat(followUp).ToList();
                var s2 = new SequenceState($"{tag}-2", prompt2, 96, cfg.BlockSize, SamplingConfig.Greedy,
                    cacheScope: tag);
                var (c2, out2) = await DrainAsync(engine.SubmitRequest(s2));
                return (out1, out2, s1, s2, c2);
            }

            var plain = await ConversationAsync("plain");
            Assert.Null(plain.seq2.SpecStats);

            engine.UpdateSpeculation(new SpeculationOptions
            {
                Enabled = true,
                SpeculatorName = SpeculatorRegistry.NGram,
                MaxDraftTokens = 3,   // Qwen's preferred window (the app takes it from its environment)
            });
            var spec = await ConversationAsync("spec");

            _output.WriteLine($"turn 2 reused {spec.c2.PrefixCacheReusedTokens} of {spec.seq2.PromptTokens.Count} prompt tokens");
            _output.WriteLine($"plain: {model.Tokenizer.Decode(plain.t2)}");
            _output.WriteLine($"spec : {model.Tokenizer.Decode(spec.t2)}");
            if (spec.seq2.SpecStats != null)
                _output.WriteLine($"verify={spec.seq2.SpecStats.VerifySteps} plain={spec.seq2.SpecStats.PlainSteps} drafted={spec.seq2.SpecStats.TokensDrafted} accepted={spec.seq2.SpecStats.TokensAccepted} "
                    + $"governor wins={spec.seq2.SpecStats.GovernorWins} losses={spec.seq2.SpecStats.GovernorLosses} parked={spec.seq2.SpecStats.GovernorParkedSteps} plainMs={spec.seq2.SpecStats.PlainMsPerToken:F1} specMs={spec.seq2.SpecStats.SpecMsPerToken:F1}");
            if (spec.seq1?.SpecStats != null)
                _output.WriteLine($"turn 1: verify={spec.seq1.SpecStats.VerifySteps} plain={spec.seq1.SpecStats.PlainSteps} drafted={spec.seq1.SpecStats.TokensDrafted} accepted={spec.seq1.SpecStats.TokensAccepted} "
                    + $"governor wins={spec.seq1.SpecStats.GovernorWins} losses={spec.seq1.SpecStats.GovernorLosses} parked={spec.seq1.SpecStats.GovernorParkedSteps}");

            Assert.Equal(plain.t1, spec.t1);
            Assert.Equal(plain.t2, spec.t2);
            // The whole retained holder, no rewind: turn one's last sampled token was never forwarded, so the
            // holder's state ends one token before the 24 it produced.
            Assert.Equal(turn1.Length + 23, spec.c2.PrefixCacheReusedTokens);
            Assert.NotNull(spec.seq2.SpecStats);
            Assert.True(spec.seq2.SpecStats.VerifySteps > 0, "speculation never verified a window on the holder");
            Assert.True(spec.seq2.SpecStats.TokensAccepted > 0, "no drafted token was accepted while quoting the passage");
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", prevRetained);
            Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", prevPerSeq);
        }
    }

    private static async Task<(InferenceCompletion completion, List<int> output)> DrainAsync(InferenceRequestHandle handle)
    {
        var output = new List<int>();
        await foreach (var t in handle.Tokens.ReadAllAsync())
            output.Add(t);
        var completion = await handle.Completion;
        return (completion, output);
    }
}
