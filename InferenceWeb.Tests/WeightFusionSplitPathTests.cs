// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Declining a load-time weight fusion must not change what the model says.
///
/// <para>
/// Fusing <c>ffn_gate</c> with <c>ffn_up</c> (and, for a mixture of experts, each
/// expert's pair) is a COPY: a GGUF writes a block's tensors alphabetically, so the
/// two halves are never byte-adjacent and joining them duplicates already-mapped
/// bytes into anonymous memory. MEASURED: 3,108 MB on gemma-4-12b UD-Q4_K_XL, 2,231 MB
/// on gemma-4-E4B Q8_0, 6,455 MB on gpt-oss-20b Q8_0. On a phone every one of those
/// bytes is charged against the jetsam limit, which is what killed TensorAgent on the
/// 12B, so iOS declines the copy and runs the two projections separately.
/// </para>
/// <para>
/// The risk that buys is a SECOND code path through the FFN, taken only on the
/// platform nobody develops on. This pins it: the same prompt, the same greedy
/// decode, both policies, in one process — which is why
/// <c>ModelBase.AllowWeightFusionCopies</c> reads its environment variable per call
/// instead of caching it.
/// </para>
/// <para>
/// Not bit-identical by construction, and the assertion says so: one matmul over a
/// [2*ff, hidden] weight and two over [ff, hidden] halves are the same arithmetic in
/// a different tile order, so a token at a near-tie can land either way. Structural
/// divergence — a wrong weight, a wrong half, an unbound tensor — does not produce a
/// near-tie, it produces an immediate mismatch.
/// </para>
/// </summary>
public sealed class WeightFusionSplitPathTests
{
    private const string FusionVar = "TS_WEIGHT_FUSION_COPIES";

    private readonly ITestOutputHelper _output;

    public WeightFusionSplitPathTests(ITestOutputHelper output) => _output = output;

    /// <summary>Gemma 4's dense FFN: the fused decode and verify graphs take the pair.</summary>
    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4|gemma4")]
    public void Gemma4AnswersTheSameWhetherOrNotItsGateAndUpWereFused() =>
        AssertSameAnswerBothWays("gemma-4|gemma4", "The capital of France is");

    /// <summary>GPT-OSS's per-expert pair: every GGML path already read the stacked pair.</summary>
    [ModelFact("TS_TEST_MODEL_DIR", "gpt-oss|gpt_oss|gptoss")]
    public void GptOssAnswersTheSameWhetherOrNotItsExpertGateAndUpWereFused() =>
        AssertSameAnswerBothWays("gpt-oss|gpt_oss|gptoss", "The capital of France is");

    private void AssertSameAnswerBothWays(string ggufContains, string prompt)
    {
        string directory = Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR");
        string path = File.Exists(directory) ? directory : TestGates.FindGguf(directory, ggufContains);
        Assert.NotNull(path);

        const int tokens = 24;
        int[] fused = Generate(path, prompt, tokens, allowFusionCopies: true);
        int[] split = Generate(path, prompt, tokens, allowFusionCopies: false);

        int matching = 0;
        while (matching < fused.Length && matching < split.Length && fused[matching] == split[matching])
            matching++;

        // A reply that ends its turn before the budget is judged against the longer
        // of the two streams: two identical 8-token answers agree completely, while a
        // stream that stops where the other continues still counts as divergence.
        int compared = Math.Min(tokens, Math.Max(fused.Length, split.Length));
        _output.WriteLine($"[fusion] {Path.GetFileName(path)}: {matching}/{compared} tokens agree (budget {tokens})");
        _output.WriteLine($"[fusion] fused={string.Join(",", fused)}");
        _output.WriteLine($"[fusion] split={string.Join(",", split)}");

        // A prefix, not the whole stream: the two orders of the same arithmetic drift.
        // Structural breakage shows up in the first token or two, never at token 20.
        Assert.True(compared > 0, "neither fusion policy generated a token");
        Assert.True(matching >= compared * 3 / 4,
            $"only {matching} of {compared} tokens agree between the fused and split FFN paths; "
            + "that is structural divergence, not floating-point drift");
    }

    /// <summary>Greedy tokens for one prompt, under one fusion policy.</summary>
    private static int[] Generate(string modelPath, string prompt, int tokens, bool allowFusionCopies)
    {
        string previous = Environment.GetEnvironmentVariable(FusionVar);
        Environment.SetEnvironmentVariable(FusionVar, allowFusionCopies ? "1" : "0");
        try
        {
            // The pinned GGML backend: one backend per process (GgmlBackendTestInitializer).
            using ModelBase model = ModelBase.Create(modelPath, TestGates.PinnedGgmlBackend);
            var config = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 1,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 64,
                BlockSize = 256,
                EnablePrefixCaching = false,
                DecodeQuantumTokens = 256,
            };
            using var engine = new InferenceEngine(model, config, NullLogger.Instance);

            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            List<int> ids = renderer.RenderToTokens(
                model.Tokenizer, model.Config?.ChatTemplate,
                new List<ChatMessage> { new() { Role = "user", Content = prompt } },
                model.Config?.Architecture ?? string.Empty,
                addGenerationPrompt: true, tools: null, enableThinking: false);

            var sequence = new SequenceState(
                "fusion-" + Guid.NewGuid().ToString("N"), ids, tokens, config.BlockSize, SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(sequence);
            var produced = new List<int>(tokens);
            try
            {
                foreach (int token in handle.Tokens.ReadAllAsync().ToBlockingEnumerable())
                    produced.Add(token);
            }
            catch (Exception) { /* the assertion reads what arrived */ }
            InferenceCompletion completion = handle.Completion.GetAwaiter().GetResult();
            // A short stream is only an answer when the model ended its turn; an
            // error or abort must not pass as agreement.
            if (produced.Count < tokens)
                Assert.Equal("eos", completion.FinishReason);
            return produced.ToArray();
        }
        finally
        {
            Environment.SetEnvironmentVariable(FusionVar, previous);
        }
    }
}
