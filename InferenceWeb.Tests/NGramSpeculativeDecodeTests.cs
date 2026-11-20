// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end proof that the algorithm layer is decoupled from the model layer:
// a model that implements ONLY ISpeculativeTarget — no draft head, no
// speculator weights, nothing a checkpoint has to ship — is driven through the
// SAME SpeculativeExecution loop that serves a NextN/MTP head, and the emitted
// stream is identical to plain decoding.
using System;
using System.Collections.Generic;
using System.Linq;

namespace InferenceWeb.Tests;

public class NGramSpeculativeDecodeTests
{
    private const int VocabSize = 64;

    [Fact]
    public void NGramSpeculation_OnAModelWithNoDraftHead_MatchesPlainGreedyExactly()
    {
        // The trunk's continuation is periodic (the model "quotes" itself), which
        // is the regime prompt-lookup decoding exists for: summarizing, editing
        // or answering about a document, repetitive structured output, agentic
        // loops. On free-form text it simply finds nothing and every step
        // degrades to a plain decode.
        var plainModel = new PeriodicTrunk(period: 7);
        var plain = PlainGreedy(plainModel, promptLen: 12, count: 40);

        var specModel = new PeriodicTrunk(period: 7);
        var exec = NewNGramExec(specModel, maxDraftTokens: 8);
        var speculative = SpeculativeGreedy(specModel, exec, promptLen: 12, count: 40);

        Assert.Equal(plain, speculative);
        Assert.True(exec.Stats.TokensAccepted > 0,
            $"n-gram never had a draft accepted (drafted={exec.Stats.TokensDrafted})");
        // The whole point: fewer trunk passes for the same tokens.
        Assert.True(exec.Stats.VerifySteps + exec.Stats.PlainSteps < 40,
            "speculation took no fewer trunk passes than plain decoding");
        Assert.Empty(specModel.ProtocolViolations);
    }

    [Fact]
    public void NGramSpeculation_NonRepeatingTrunk_StillMatchesPlainGreedy()
    {
        // No suffix ever repeats, so every step degrades to a plain decode. The
        // stream must be identical and nothing must break.
        var plainModel = new PeriodicTrunk(period: 0);
        var plain = PlainGreedy(plainModel, promptLen: 8, count: 24);

        var specModel = new PeriodicTrunk(period: 0);
        var exec = NewNGramExec(specModel, maxDraftTokens: 8);
        var speculative = SpeculativeGreedy(specModel, exec, promptLen: 8, count: 24);

        Assert.Equal(plain, speculative);
        Assert.Empty(specModel.ProtocolViolations);
    }

    [Fact]
    public void NGramSpeculation_NeedsNoHiddenStateCapture()
    {
        // A weight-free speculator lets the loop skip the trunk's hidden-state
        // tap entirely — the model must never be asked for one.
        var model = new PeriodicTrunk(period: 5);
        var exec = NewNGramExec(model, maxDraftTokens: 8);
        SpeculativeGreedy(model, exec, promptLen: 10, count: 30);

        Assert.Equal(0, model.HiddenStateRequests);
    }

    [Theory]
    [InlineData(8, false, true, true)]
    [InlineData(64, false, true, true)]
    [InlineData(64, true, true, true)]
    [InlineData(64, true, false, true)]
    [InlineData(8, false, true, false)]
    [InlineData(8, false, false, true, 7)] // Qwen: seven drafts plus the anchor are eight verify rows.
    [InlineData(64, true, false, true, 7)]
    public void HardTargetLimitBoundsExplicitAndDirectAlgorithmsAcrossRollback(
        int requestedDrafts, bool directAlgorithm, bool persistsAcceptedKv, bool explicitRequest, int targetLimit = 5)
    {
        var plainModel = new PeriodicTrunk(period: 7) { MismatchPosition = 35 };
        List<int> plain = PlainGreedy(plainModel, promptLen: 12, count: 96);
        var model = new PeriodicTrunk(period: 7)
        {
            SpecMaxDraftTokens = targetLimit,
            SpecPreferredNGramDraftWindow = 64,
            SpecVerifyPersistsAcceptedKv = persistsAcceptedKv,
            MismatchPosition = 35,
        };
        ISpeculator algorithm = directAlgorithm ? new NGramSpeculator(requestedDrafts)
            : SpeculatorRegistry.Create(model, new SpeculationOptions
            {
                Enabled = true,
                SpeculatorName = SpeculatorRegistry.NGram,
                MaxDraftTokens = requestedDrafts,
                MaxDraftTokensExplicit = explicitRequest,
            }, out _)!;
        Assert.NotNull(algorithm);
        if (!directAlgorithm) Assert.Equal(targetLimit, algorithm.MaxDraftTokens);
        using var execution = new SpeculativeExecution(model, algorithm) { AdaptiveSpeculation = false };

        List<int> actual = SpeculativeGreedy(model, execution, promptLen: 12, count: 96);

        Assert.Equal(targetLimit, execution.MaxDraftTokens);
        Assert.Equal(plain, actual);
        Assert.Equal(plainModel.CommittedTokens, model.CommittedTokens);
        Assert.Contains(targetLimit + 1, model.VerifyWidths);
        Assert.All(model.VerifyWidths, width => Assert.InRange(width, 2, targetLimit + 1));
        Assert.True(execution.Stats.TokensAccepted > 0);
        Assert.True(execution.Stats.RollbackSteps > 0);
        if (persistsAcceptedKv) Assert.Empty(model.ReplayWidths);
        else
        {
            Assert.NotEmpty(model.ReplayWidths);
            Assert.All(model.ReplayWidths, width => Assert.InRange(width, 1, targetLimit));
        }
        Assert.Empty(model.ProtocolViolations);
    }

    /// <summary>
    /// On the batched solo step (<see cref="SpeculativeExecution.BeginSoloBatchStep"/>), a step whose n-gram lookup finds
    /// nothing is a plain governor sample, and a step that drafts is a speculative one. The matchless step paid for no head
    /// and verifies its anchor alone, which on the unified trunk is the plain decode step. Counted as speculative, such steps
    /// diluted every probe of free prose toward plain cost and won it: Muse-Glimmer-30B's n-gram took 36 verify windows over a
    /// 512-token answer and ran 5% under plain on ggml_cuda.
    /// </summary>
    [Fact]
    public void ABatchedSoloStepWhoseLookupFindsNothingIsAPlainSampleAndADraftingOneASpeculativeOne()
    {
        var model = new PeriodicTrunk(period: 0);
        using var exec = new SpeculativeExecution(model, new NGramSpeculator(8));
        // Distinct tokens: no suffix of any length has occurred before.
        var committed = Enumerable.Range(10, 12).ToList();
        exec.SeedCommitted(committed.ToArray());
        var drafts = new List<int>();
        for (int i = 0; i < 4; i++)
        {
            int anchor = 40 + i, position = committed.Count;
            Assert.True(exec.BeginSoloBatchStep());
            Assert.Equal(0, exec.ProposeForBatch(anchor, position, 4, drafts));
            exec.CommitBatch(anchor, position, drafts, 0, null, 0, verifyTicks: 1_000);
            exec.EndSoloBatchStep(1, 1_000);
            committed.Add(anchor);
        }
        Assert.True(exec.Governor.PlainMsPerToken > 0, "the matchless steps were not plain samples");
        Assert.Equal(0, exec.Governor.SpecMsPerToken);

        // Now the stream repeats 40, 41, 42, 43: from 41 on each suffix occurred before, the lookup drafts, and the step is a
        // speculative sample (the first one is not measured).
        for (int anchor = 40; anchor <= 43; anchor++)
        {
            int position = committed.Count;
            Assert.True(exec.BeginSoloBatchStep());
            int proposed = exec.ProposeForBatch(anchor, position, 4, drafts);
            Assert.True(anchor == 40 ? proposed == 0 : proposed > 0, $"anchor {anchor}: {proposed} drafts");
            exec.CommitBatch(anchor, position, drafts, 0, null, 0, verifyTicks: 4_000);
            exec.EndSoloBatchStep(1, 4_000);
            committed.Add(anchor);
        }
        Assert.True(exec.Governor.SpecMsPerToken > 0, "the drafting steps were not speculative samples");
    }

    [Fact]
    public void ZeroVerificationCapacityDeclinesRegistryAndDirectExecutionRunsPlainly()
    {
        var model = new PeriodicTrunk(period: 7) { SpecMaxDraftTokens = 0 };
        Assert.Null(SpeculatorRegistry.Create(model, new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.NGram,
            MaxDraftTokens = 64, MaxDraftTokensExplicit = true,
        }, out string decline));
        Assert.Contains("no capacity", decline);
        using var execution = new SpeculativeExecution(model, new NGramSpeculator(64))
        {
            AdaptiveSpeculation = false,
        };
        Assert.Equal(PlainGreedy(new PeriodicTrunk(period: 7), 12, 40),
            SpeculativeGreedy(model, execution, 12, 40));
        Assert.Equal(0, execution.MaxDraftTokens);
        Assert.Empty(model.VerifyWidths);
    }

    [Fact]
    public void AlgorithmExceedingRequestedWindowIsRejectedBeforeTrunkMutation()
    {
        var model = new PeriodicTrunk(period: 7) { SpecMaxDraftTokens = 5 };
        using var execution = new SpeculativeExecution(model, new OversizedSpeculator())
        {
            AdaptiveSpeculation = false,
        };
        int pending = Argmax(execution.PrefillStep(Enumerable.Range(1, 12).ToArray(), 0));
        int[] committed = model.CommittedTokens.ToArray();
        var error = Assert.Throws<InvalidOperationException>(() =>
            execution.DecodeStep(pending, 12, 64, Argmax));
        Assert.Contains("limited to 5 drafts", error.Message);
        Assert.Empty(model.VerifyWidths);
        Assert.Equal(committed, model.CommittedTokens);
    }

    private sealed class OversizedSpeculator : ISpeculator
    {
        public string Name => "oversized-test";
        public int MaxDraftTokens => 64;
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0;
        public bool NeedsHiddenState => false;
        public bool HandlesOwnPrefill => false;
        public int Propose(in DraftContext context, List<int> draftOut)
        {
            for (int i = 0; i <= context.MaxTokens; ++i) draftOut.Add(1);
            return draftOut.Count;
        }
        public void Commit(int[] tokens, float[] hRows, int startPos) { }
        public void Reset() { }
        public void Dispose() { }
    }

    private static SpeculativeExecution NewNGramExec(PeriodicTrunk model, int maxDraftTokens)
    {
        var speculator = SpeculatorRegistry.Create(
            model,
            new SpeculationOptions
            {
                Enabled = true,
                SpeculatorName = SpeculatorRegistry.NGram,
                MaxDraftTokens = maxDraftTokens,
            },
            out string decline);
        Assert.True(speculator != null, decline);
        var exec = new SpeculativeExecution(model, speculator);
        // Measure the algorithm, not the governor: on a fake model every step
        // costs the same, so the cost governor's verdict is noise.
        exec.AdaptiveSpeculation = false;
        return exec;
    }

    private static List<int> PlainGreedy(PeriodicTrunk model, int promptLen, int count)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = model.Forward(prompt);
        var output = new List<int>();
        int t = Argmax(logits);
        output.Add(t);
        for (int i = 1; i < count; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            output.Add(t);
        }
        return output;
    }

    private static List<int> SpeculativeGreedy(PeriodicTrunk model, SpeculativeExecution exec,
        int promptLen, int count)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = exec.PrefillStep(prompt, 0);

        var output = new List<int>();
        int last = Argmax(logits);
        output.Add(last);
        int pos = promptLen;

        while (output.Count < count)
        {
            var accepted = new List<int>();
            var outcome = exec.DecodeStep(last, pos, count - output.Count - 1,
                drawNext: Argmax, onDraftAccepted: accepted.Add);

            if (!outcome.UsedSpeculation)
            {
                last = Argmax(outcome.NextLogits);
                pos += 1;
                output.Add(last);
                continue;
            }

            pos += outcome.AcceptedCount + 1;
            last = outcome.NextToken;
            foreach (int d in accepted)
            {
                if (output.Count >= count) return output;
                output.Add(d);
            }
            if (output.Count < count)
                output.Add(outcome.NextToken);
        }
        return output;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>
    /// A trunk with no draft head whatsoever. Its next-token function repeats
    /// with <c>period</c> (0 = never repeats), so an n-gram drafter can find the
    /// continuation in the context exactly the way it does on real repetitive
    /// text. Records any request for hidden states, which a weight-free
    /// speculator must never trigger.
    /// </summary>
    private sealed class PeriodicTrunk : ISpeculativeTarget
    {
        private readonly int _period;
        private readonly List<int> _trunk = new();

        public PeriodicTrunk(int period) => _period = period;

        public List<string> ProtocolViolations { get; } = new();
        public int HiddenStateRequests { get; private set; }
        public int SpecMaxDraftTokens { get; init; } = int.MaxValue;
        public int SpecPreferredNGramDraftWindow { get; init; }
        public bool SpecVerifyPersistsAcceptedKv { get; init; }
        public int? MismatchPosition { get; init; }
        public IReadOnlyList<int> CommittedTokens => _trunk;
        public List<int> VerifyWidths { get; } = new();
        public List<int> ReplayWidths { get; } = new();
        private bool _replaying;

        /// <summary>The trunk's argmax for the row of the token at position p.</summary>
        private int Next(int p) => p == MismatchPosition ? 50 : _period > 0
            ? 3 + (p % _period)
            : 3 + ((p * 13 + 7) % (VocabSize - 4));

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize, HiddenSize = 4 };
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public void TruncateKVCache(int tokenCount) { }
        public void ResetKVCache() => _trunk.Clear();
        public void Dispose() { }

        public float[] Forward(int[] tokens)
        {
            int start = _trunk.Count;
            _trunk.AddRange(tokens);
            var logits = new float[VocabSize];
            logits[Next(start + tokens.Length - 1)] = 10f;
            return logits;
        }

        public int CacheSeqLen => _trunk.Count;
        public int MaxContextLength => 4096;

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (allLogitsRows)
            {
                VerifyWidths.Add(tokens.Length);
                Assert.True(tokens.Length - 1 <= SpecMaxDraftTokens, "Verification exceeded the target's cache capacity.");
            }
            else if (_replaying)
            {
                ReplayWidths.Add(tokens.Length);
                _replaying = false;
            }
            if (hAllOut != null)
                HiddenStateRequests++;
            int start = _trunk.Count;
            _trunk.AddRange(tokens);
            if (allLogitsRows)
            {
                for (int i = 0; i < tokens.Length; i++)
                {
                    Array.Clear(logitsOut, i * VocabSize, VocabSize);
                    logitsOut[i * VocabSize + Next(start + i)] = 10f;
                }
            }
            else
            {
                Array.Clear(logitsOut, 0, VocabSize);
                logitsOut[Next(start + tokens.Length - 1)] = 10f;
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen) { }
        public void SpecSnapshotRecurrentState() { }
        public void SpecRestoreRecurrentState() => _replaying = true;

        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _trunk.Count)
            {
                ProtocolViolations.Add($"rewind to {length} outside [0, {_trunk.Count}]");
                return;
            }
            _trunk.RemoveRange(length, _trunk.Count - length);
        }
    }
}
