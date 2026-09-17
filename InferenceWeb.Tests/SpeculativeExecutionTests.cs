// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Unit tests for NextN/MTP speculative decoding: the shared
// SpeculativeExecution core (draft / verify / rollback / catch-up protocol
// against a deterministic fake model — no GGUF needed) and the engine path
// (BatchExecutor routing, ExtraTokens streaming, block-boundary draft capping,
// EOS inside an accepted draft window).
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public class SpeculativeExecutionTests
{
    private const int VocabSize = 64;
    private const int HiddenSize = 4;
    private const int BlockSize = 8;

    // ----- shared-core tests (drive SpeculativeExecution directly) -----

    [Fact]
    public void DecodeStep_PerfectDrafts_AcceptsWholeWindowAndReturnsBonus()
    {
        var model = new FakeSpeculativeModel();
        var exec = NewExec(model, maxDraftTokens: 4);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);

        var accepted = new List<int>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 4,
            drawNext: Argmax, onDraftAccepted: accepted.Add);

        Assert.True(outcome.UsedSpeculation);
        Assert.Equal(4, outcome.AcceptedCount);
        Assert.Equal(new[] { model.ExpectedNext(pos), model.ExpectedNext(pos + 1),
            model.ExpectedNext(pos + 2), model.ExpectedNext(pos + 3) }, accepted);
        // Bonus row: the trunk's prediction one past the accepted window.
        Assert.Equal(model.ExpectedNext(pos + 4), outcome.NextToken);
        Assert.Equal(pos + 5, model.CacheSeqLen); // 1 + 4 drafts advanced
        Assert.Equal(4, exec.Stats.TokensDrafted);
        Assert.Equal(4, exec.Stats.TokensAccepted);
        Assert.Equal(0, exec.Stats.RollbackSteps);
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void DecodeStep_WrongDraft_RollsBackRecurrentStateAndCorrects()
    {
        var model = new FakeSpeculativeModel();
        var exec = NewExec(model, maxDraftTokens: 4);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        model.DraftWrongPositions.Add(pos + 1); // second draft is wrong

        var accepted = new List<int>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 4,
            drawNext: Argmax, onDraftAccepted: accepted.Add);

        Assert.True(outcome.UsedSpeculation);
        Assert.Equal(1, outcome.AcceptedCount);
        Assert.Equal(new[] { model.ExpectedNext(pos) }, accepted);
        // The drawn token at the mismatching row IS the correction.
        Assert.Equal(model.ExpectedNext(pos + 1), outcome.NextToken);
        Assert.Equal(pos + 2, model.CacheSeqLen); // 1 + 1 accepted
        Assert.Equal(1, exec.Stats.RollbackSteps);
        Assert.Equal(1, model.SnapshotCalls);
        Assert.Equal(1, model.RestoreCalls);
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void DecodeStep_LowConfidenceDrafts_DegradesToPlainStep()
    {
        var model = new FakeSpeculativeModel();
        var exec = NewExec(model, maxDraftTokens: 4);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        model.LowConfidencePositions.Add(pos); // first draft already unconfident

        var outcome = exec.DecodeStep(lastToken, pos, kMax: 4, drawNext: Argmax);

        Assert.False(outcome.UsedSpeculation);
        Assert.Equal(0, outcome.AcceptedCount);
        Assert.Equal(-1, outcome.NextToken);
        Assert.Equal(model.ExpectedNext(pos), Argmax(outcome.NextLogits));
        Assert.Equal(pos + 1, model.CacheSeqLen);
        Assert.Equal(1, exec.Stats.PlainSteps);
        Assert.Equal(0, exec.Stats.TokensDrafted);
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void DecodeStep_AdjustDraftLogitsHook_SeesGrowingPendingWindowAndRedirectsDrafts()
    {
        var model = new FakeSpeculativeModel();
        var exec = NewExec(model, maxDraftTokens: 3);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);

        // The hook must fire once per draft step with the drafts pending so
        // far — that's the contract penalty-aligned drafting relies on.
        var pendingSnapshots = new List<int[]>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 3,
            drawNext: Argmax,
            adjustDraftLogits: (logits, pending) => pendingSnapshots.Add(pending.ToArray()),
            onDraftAccepted: _ => { });

        Assert.Equal(3, outcome.AcceptedCount);
        Assert.Equal(3, pendingSnapshots.Count);
        Assert.Empty(pendingSnapshots[0]);
        Assert.Equal(new[] { model.ExpectedNext(pos) }, pendingSnapshots[1]);
        Assert.Equal(new[] { model.ExpectedNext(pos), model.ExpectedNext(pos + 1) }, pendingSnapshots[2]);

        // And when the hook rewrites the draft distribution (the way sampler
        // penalties do), the REWRITTEN argmax is what gets verified: divert
        // the first draft to a token the trunk disagrees with -> rejection.
        model.Reset();
        exec = NewExec(model, maxDraftTokens: 3);
        pos = PrefillPrompt(model, exec, promptLen: 5, out lastToken);
        int diverted = (model.ExpectedNext(pos) + 1) % VocabSize;
        var outcome2 = exec.DecodeStep(lastToken, pos, kMax: 3,
            drawNext: Argmax,
            adjustDraftLogits: (logits, pending) =>
            {
                if (pending.Count == 0)
                    logits[diverted] = logits[model.ExpectedNext(pos)] + 5f;
            });
        Assert.Equal(0, outcome2.AcceptedCount);
        Assert.Equal(model.ExpectedNext(pos), outcome2.NextToken);
    }

    // ----- engine-path tests (InferenceEngine + BatchExecutor routing) -----

    [Fact]
    public void EngineSpec_GreedyStream_MatchesPlainDecodeAcrossBlockBoundaries()
    {
        // BlockSize 8 with 40 generated tokens crosses several block
        // boundaries; the executor must cap each draft window to the blocks
        // the scheduler reserved (boundary steps degrade to plain decode)
        // instead of overrunning the block table. A couple of wrong drafts
        // exercise rollback inside the engine flow. The output stream must
        // still equal the model's deterministic plain-greedy chain exactly.
        const int promptLen = 5;
        const int maxNew = 40;

        var model = new FakeSpeculativeModel();
        model.DraftWrongPositions.Add(promptLen + 7);
        model.DraftWrongPositions.Add(promptLen + 19);

        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true);

        Assert.Equal(SequenceStatus.FinishedLengthCapped, seq.Status);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.NotNull(seq.SpecStats);
        Assert.True(seq.SpecStats.TokensAccepted > 0, "speculation never accepted a draft");
        Assert.True(seq.SpecStats.RollbackSteps >= 1, "wrong drafts should have caused a rollback");
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void EngineSpec_EosInsideAcceptedWindow_StopsAndTruncatesTrailingDrafts()
    {
        const int promptLen = 5;
        var model = new FakeSpeculativeModel();
        // The 6th generated token is EOS; with an 8-token draft window it
        // lands inside an accepted speculative batch.
        int eosToken = model.ExpectedNext(promptLen + 4);
        model.EosTokenId = eosToken;

        var seq = RunEngineRequest(model, promptLen, maxNewTokens: 32, specEnabled: true,
            beforeDispose: () => Assert.Equal(promptLen + 6, model.CacheSeqLen));

        Assert.Equal(SequenceStatus.FinishedStopped, seq.Status);
        // Output ends at EOS (kept, per the engine's contract) with the
        // speculative acceptances past it truncated.
        Assert.Equal(ExpectedChain(model, promptLen, 5), seq.OutputTokens.Take(5));
        Assert.Equal(eosToken, seq.OutputTokens[^1]);
        Assert.Equal(6, seq.OutputTokens.Count);
        // Nothing was forwarded past the stop: the trunk holds exactly the prompt
        // and the six emitted tokens, so a holder retained from here matches the
        // tokens it is recorded as holding (and, on a sliding-window ring, no
        // post-stop row evicted a position the next turn still attends to).
        // The cache-length assertion above runs before engine shutdown clears state.
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Theory]
    [InlineData(false)] // linear trunk
    [InlineData(true)]  // batched (paged) trunk
    public void EngineSpec_PrefixCaching_EosPastBlockBoundary_FinishesWithoutThrow(bool batchedTrunk)
    {
        // Production repro (the server enables prefix caching by default): a
        // long speculative generation crosses a generated block boundary, then
        // EOS lands inside an accepted speculative batch. The step advanced the
        // committed-token count over the whole batch, so after the engine
        // truncates the trailing drafts the committed count outruns the
        // surviving token list. FinishSequence -> CacheFullBlocksForSequence
        // used to hash positions past the end of that list and throw
        // ArgumentOutOfRangeException(pos) — the crash this change fixes.
        const int promptLen = 8; // fills block 0 exactly
        var model = new FakeSpeculativeModel { BatchedTrunkEnabled = batchedTrunk };
        // EOS = the 7th generated token (output index 6). Its committed
        // position (15) sits just below the block-1 boundary (16), so the
        // accepted draft tail pushes the committed count to/past 16 while the
        // surviving output stops at 15 -> committed > total at finish time.
        int eosToken = model.ExpectedNext(promptLen - 1 + 6);
        model.EosTokenId = eosToken;

        var seq = RunEngineRequest(model, promptLen, maxNewTokens: 64,
            specEnabled: true, enablePrefixCaching: true);

        Assert.Equal(SequenceStatus.FinishedStopped, seq.Status);
        Assert.Equal(7, seq.OutputTokens.Count);
        Assert.Equal(eosToken, seq.OutputTokens[^1]);
        Assert.Equal(ExpectedChain(model, promptLen, 6), seq.OutputTokens.Take(6));
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void EngineSpec_BatchedTrunk_GreedyStream_MatchesPlainDecodeAcrossBlockBoundaries()
    {
        // Batched-trunk speculation: every trunk pass must go through
        // SpecForwardBatched (the linear SpecForward stays untouched), the
        // sequence's K/V is flagged as paged storage, rollbacks use the
        // per-slot snapshot APIs — and the output stream still equals the
        // deterministic plain-greedy chain exactly.
        const int promptLen = 5;
        const int maxNew = 40;

        var model = new FakeSpeculativeModel { BatchedTrunkEnabled = true };
        model.DraftWrongPositions.Add(promptLen + 7);
        model.DraftWrongPositions.Add(promptLen + 19);

        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true);

        Assert.Equal(SequenceStatus.FinishedLengthCapped, seq.Status);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.NotNull(seq.SpecStats);
        Assert.True(seq.SpecStats.TokensAccepted > 0, "speculation never accepted a draft");
        Assert.True(seq.SpecStats.RollbackSteps >= 1, "wrong drafts should have caused a rollback");
        Assert.True(model.BatchedSpecForwardCalls > 0, "spec trunk never used the batched path");
        Assert.Equal(0, model.LinearSpecForwardCalls);
        Assert.True(model.SlotSnapshotCalls > 0, "verify never snapshotted the slot state");
        Assert.True(model.SlotRestoreCalls >= 1, "rollback never restored the slot state");
        Assert.True(seq.KvStateInPagedStorage, "batched-trunk steps must mark K/V as paged");
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void EngineSpec_BatchedTrunk_EosInsideAcceptedWindow_StopsAndTruncates()
    {
        const int promptLen = 5;
        var model = new FakeSpeculativeModel { BatchedTrunkEnabled = true };
        int eosToken = model.ExpectedNext(promptLen + 4);
        model.EosTokenId = eosToken;

        var seq = RunEngineRequest(model, promptLen, maxNewTokens: 32, specEnabled: true);

        Assert.Equal(SequenceStatus.FinishedStopped, seq.Status);
        Assert.Equal(ExpectedChain(model, promptLen, 5), seq.OutputTokens.Take(5));
        Assert.Equal(eosToken, seq.OutputTokens[^1]);
        Assert.Equal(6, seq.OutputTokens.Count);
        Assert.True(model.BatchedSpecForwardCalls > 0);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void EngineSpec_Disabled_ProducesSameStreamWithoutSpeculation()
    {
        const int promptLen = 5;
        const int maxNew = 16;
        var model = new FakeSpeculativeModel();

        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: false);

        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.Null(seq.SpecStats);
    }

    // --mtp-spec is requested but the backend can't drive the accelerated MTP
    // path (SpeculationProfitable=false, e.g. the pure-C# CUDA backend for
    // Gemma 4). The engine must serve the fast standard decode — never the
    // net-negative per-op spec path — so speculation never engages: no
    // SpecForward calls (linear OR batched) and no spec stats, with output
    // identical to the plain stream. Regression guard for the backend gate.
    [Theory]
    [InlineData(false)]   // linear-trunk-capable model
    [InlineData(true)]    // batched-trunk-capable model
    public void EngineSpec_UnprofitableBackend_FallsBackToStandardDecode(bool batchedTrunk)
    {
        const int promptLen = 5;
        const int maxNew = 16;
        var model = new FakeSpeculativeModel
        {
            SpeculationProfitable = false,
            BatchedTrunkEnabled = batchedTrunk,
        };

        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true);

        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.Equal(0, model.LinearSpecForwardCalls);
        Assert.Equal(0, model.BatchedSpecForwardCalls);
        Assert.Null(seq.SpecStats);
    }

    // ----- helpers -----

    /// <summary>Build the shared core over whatever algorithm the fake's draft
    /// head implies — the same resolution the engine and the CLI go through.</summary>
    [Fact]
    public void EngineSpec_NGramArmedAfterAPrefixReuse_IsSeededWithTheReusedTokens()
    {
        // A follow-up whose prompt reuses cached blocks arms at a prefill chunk that
        // starts past position 0. The n-gram drafter's corpus used to be empty
        // there, its first commit looked like a gap, and it stayed silent for the
        // whole request. The fake's stream is periodic in position (period 64), so
        // a seeded corpus holding the first request's output must yield drafts.
        const int promptLen = 100;
        const int maxNew = 70;
        var model = new FakeSpeculativeModel();
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 64,
            SoloPrefillChunkSize = 8192,
            NumBlocks = 64,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 1,
            Speculation = new SpeculationOptions
            {
                Enabled = true,
                SpeculatorName = SpeculatorRegistry.NGram,
                MaxDraftTokens = 4,
            },
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var greedy = new SamplingConfig { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1f, PresencePenalty = 0f, FrequencyPenalty = 0f };

        var first = new SequenceState("reuse-1", Enumerable.Range(1, promptLen).ToList(), maxNew, BlockSize, greedy);
        engine.SubmitRequest(first).Completion.GetAwaiter().GetResult();
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), first.OutputTokens);

        var prompt2 = Enumerable.Range(1, promptLen).Concat(first.OutputTokens).Concat(new[] { 3, 5 }).ToList();
        var second = new SequenceState("reuse-2", prompt2, maxNew, BlockSize, greedy);
        var completion = engine.SubmitRequest(second).Completion.GetAwaiter().GetResult();

        Assert.True(completion.PrefixCacheReusedTokens > 0, "the follow-up should have reused the first request's cache");
        Assert.Equal(ExpectedChain(model, prompt2.Count, maxNew), second.OutputTokens);
        Assert.NotNull(second.SpecStats);
        Assert.True(second.SpecStats.VerifySteps > 0, "speculation never verified a window after the prefix reuse");
        Assert.True(second.SpecStats.TokensAccepted > 0, "no draft was accepted: the corpus was not seeded with the reused tokens");
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void EngineSpec_PromptLongerThanOnePrefillChunk_StaysArmedAcrossTheChunks()
    {
        // A 200-token prompt through 64-token solo chunks is four prefill steps.
        // The executor used to re-arm speculation on every one of them, and from
        // chunk two the arm looked like a KV-prefix reuse at position 64, which a
        // per-token head declines - so the whole request decoded plainly. With the
        // phone's 1024-token chunk and a 5-7k-token agent prompt that was every
        // request TensorAgent ever made.
        const int promptLen = 200;
        const int maxNew = 24;
        var model = new FakeSpeculativeModel();
        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true, soloPrefillChunk: 64);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.NotNull(seq.SpecStats);
        Assert.True(seq.SpecStats.VerifySteps > 0,
            "speculation never verified a window: the second prefill chunk disarmed it");
        Assert.True(seq.SpecStats.TokensAccepted > 0);
        Assert.Empty(model.ProtocolViolations);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void DraftHeadSpeculator_ArmsAfterAPrefixReuse_OnlyWhenTheHeadKeepsNoState(bool resumes)
    {
        // Gemma 4's head reads the trunk's donor KV and the hidden state it is
        // handed, so it drafts from any position; a NextN block with its own KV
        // cache cannot, and the executor must keep declining it after a reuse.
        var model = new FakeSpeculativeModel { ResumesAfterGap = resumes };
        var spec = SpeculatorRegistry.Create(model,
            new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead },
            out string decline);
        Assert.Null(decline);
        Assert.Equal(resumes, spec.CanArmAfterPrefixReuse);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void PlainStep_OfAHiddenFreeSpeculator_UsesTheModelsOwnDecodeWhenTheTrunkAllows(bool viaForward)
    {
        // n-gram needs no hidden state, so a step it cannot draft for (an empty
        // corpus) is an ordinary decode - which on a trunk that says so runs the
        // model's fused decode with the LM head folded in, not a one-row
        // SpecForward with a per-op tail. The stream is the same either way.
        var model = new FakeSpeculativeModel { PlainStepUsesForward = viaForward };
        var speculator = SpeculatorRegistry.Create(model,
            new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram },
            out string decline);
        Assert.Null(decline);
        var exec = new SpeculativeExecution(model, speculator);
        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        int specCallsBefore = model.LinearSpecForwardCalls;
        int forwardBefore = model.ForwardCalls;

        var outcome = exec.DecodeStep(lastToken, pos, kMax: 4, drawNext: Argmax);

        Assert.False(outcome.UsedSpeculation);
        Assert.Equal(model.ExpectedNext(pos), Argmax(outcome.NextLogits));
        Assert.Equal(pos + 1, model.CacheSeqLen);
        Assert.Equal(viaForward ? 1 : 0, model.ForwardCalls - forwardBefore);
        Assert.Equal(viaForward ? 0 : 1, model.LinearSpecForwardCalls - specCallsBefore);
    }

    [Fact]
    public void EngineSpec_MediaTurn_ArmsAHiddenFreeSpeculatorAfterThePlainPrefill()
    {
        // An image/audio turn prefills on the plain path (only Forward's inject
        // hook can place the embeddings), and until now that was the end of it:
        // nothing armed at the first decode step, so the whole answer decoded
        // plainly. n-gram needs no hidden state, so it starts there from the tokens
        // the trunk already holds. The fake's truth stream repeats every 64
        // positions, which is what gives the lookup something to find.
        const int promptLen = 5;
        const int maxNew = 150;
        var injector = new FakeInjector { PromptTokens = promptLen };
        var model = new FakeSpeculativeModel { Injector = injector };
        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true, speculator: SpeculatorRegistry.NGram);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.True(injector.SlicesQueued > 0, "the plain prefill never queued the media embeddings");
        Assert.NotNull(seq.SpecStats);
        Assert.True(seq.SpecStats.VerifySteps > 0, "speculation never armed after the plain prefill");
        Assert.True(seq.SpecStats.TokensAccepted > 0);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void EngineSpec_MediaTurn_LeavesALearnedHeadAlone()
    {
        // A per-token head chains the trunk's hidden state of the last committed
        // token, which a plain prefill never captured: it must not arm late.
        const int promptLen = 5;
        const int maxNew = 40;
        var model = new FakeSpeculativeModel { Injector = new FakeInjector { PromptTokens = promptLen } };
        var seq = RunEngineRequest(model, promptLen, maxNew, specEnabled: true, speculator: SpeculatorRegistry.DraftHead);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.True(seq.SpecStats == null || seq.SpecStats.VerifySteps == 0);
        Assert.Empty(model.ProtocolViolations);
    }

    /// <summary>A media turn's injector: pending until the plain prefill has queued
    /// the whole prompt's slices, exactly the signal the planner reads.</summary>
    private sealed class FakeInjector : IMultimodalInjector
    {
        public int PromptTokens { get; set; }
        public int SlicesQueued { get; private set; }
        private int _queuedThrough;

        public void LoadProjectors(string mmProjPath) { }
        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens, string requestId = null) => inputTokens;
        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null) => true;
        public bool QueuePromptEmbeddingsForSlice(int promptStartToken, int tokenCount, string requestId = null)
        {
            SlicesQueued++;
            _queuedThrough = Math.Max(_queuedThrough, promptStartToken + tokenCount);
            return true;
        }
        public int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null) => reusablePrefixTokenCount;
        public int ClampTrimStart(int trimStartTokenCount, string requestId = null) => trimStartTokenCount;
        public void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null) { }
        public bool HasPendingEmbeddings(string requestId) => _queuedThrough < PromptTokens;
        public void ClearPreparedPromptState(string requestId) { }
    }

    [Fact]
    public void Engine_UpdateSpeculation_SwitchesThePolicyForTheNextRequestWithoutARebuild()
    {
        // The settings switch in the app: the engine keeps running, and the next
        // turn follows the new policy - on, off, and on again.
        const int promptLen = 5;
        const int maxNew = 24;
        var model = new FakeSpeculativeModel();
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 64,
            NumBlocks = 32,
            BlockSize = BlockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
            Speculation = SpeculationOptions.Disabled,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var greedy = new SamplingConfig { Temperature = 0f, TopK = 0, TopP = 1f, RepetitionPenalty = 1f };

        SequenceState Run(string id)
        {
            var seq = new SequenceState(id, Enumerable.Range(1, promptLen).ToList(), maxNew, BlockSize, greedy);
            engine.SubmitRequest(seq).Completion.GetAwaiter().GetResult();
            Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
            model.Reset();
            return seq;
        }

        Assert.Null(Run("off-1").SpecStats);

        engine.UpdateSpeculation(new SpeculationOptions { Enabled = true, MaxDraftTokens = 8, MinDraftProb = 0.5f });
        SequenceState on = Run("on-1");
        Assert.NotNull(on.SpecStats);
        Assert.True(on.SpecStats.VerifySteps > 0, "the new policy did not reach the next request");

        engine.UpdateSpeculation(SpeculationOptions.Disabled);
        Assert.Null(Run("off-2").SpecStats);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void SeedCommitted_WithAResumableLearnedHead_RunsOnePlainStepThenDrafts()
    {
        // Gemma 4's head keeps no state of its own, so it can start over tokens the
        // trunk already holds - but it chains the trunk hidden state of the last
        // committed token, which a plain prefill never captured. The first step is
        // therefore plain (and captures it); the second drafts from it.
        var model = new FakeSpeculativeModel { ResumesAfterGap = true };
        var speculator = SpeculatorRegistry.Create(model,
            new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead, MaxDraftTokens = 4, MinDraftProb = 0.5f },
            out string decline);
        Assert.Null(decline);
        var exec = new SpeculativeExecution(model, speculator);
        Assert.True(exec.CanSeedCommitted);

        int[] prompt = Enumerable.Range(1, 5).ToArray();
        float[] logits = model.Forward(prompt);          // the plain prefill
        int lastToken = Argmax(logits);
        exec.SeedCommitted(prompt);

        var first = exec.DecodeStep(lastToken, prompt.Length, kMax: 4, drawNext: Argmax);
        Assert.False(first.UsedSpeculation);
        Assert.Equal(model.ExpectedNext(prompt.Length), Argmax(first.NextLogits));

        int next = Argmax(first.NextLogits);
        var second = exec.DecodeStep(next, prompt.Length + 1, kMax: 4, drawNext: Argmax);
        Assert.True(second.UsedSpeculation);
        Assert.True(second.AcceptedCount > 0);
        Assert.Empty(model.ProtocolViolations);
    }

    private static SpeculativeExecution NewExec(FakeSpeculativeModel model, int maxDraftTokens,
        float? minDraftProb = null)
    {
        var options = new SpeculationOptions
        {
            Enabled = true,
            MaxDraftTokens = maxDraftTokens,
            MinDraftProb = minDraftProb,
        };
        var speculator = SpeculatorRegistry.Create(model, options, out string decline);
        Assert.True(speculator != null, decline);
        return new SpeculativeExecution(model, speculator);
    }

    private static int PrefillPrompt(FakeSpeculativeModel model, SpeculativeExecution exec, int promptLen, out int lastToken)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = exec.PrefillStep(prompt, 0);
        lastToken = Argmax(logits);
        Assert.Equal(model.ExpectedNext(promptLen - 1), lastToken);
        return promptLen;
    }

    /// <summary>The deterministic greedy stream: token i of the generation is
    /// the trunk's argmax for the row at absolute position promptLen-1+i.</summary>
    private static List<int> ExpectedChain(FakeSpeculativeModel model, int promptLen, int count)
    {
        var expected = new List<int>(count);
        for (int i = 0; i < count; i++)
            expected.Add(model.ExpectedNext(promptLen - 1 + i));
        return expected;
    }

    private static SequenceState RunEngineRequest(FakeSpeculativeModel model, int promptLen, int maxNewTokens,
        bool specEnabled, bool enablePrefixCaching = false, int soloPrefillChunk = 0, string speculator = null,
        Action beforeDispose = null)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 64,
            SoloPrefillChunkSize = soloPrefillChunk > 0 ? soloPrefillChunk : 8192,
            NumBlocks = 32,
            BlockSize = BlockSize,
            EnablePrefixCaching = enablePrefixCaching,
            DecodeQuantumTokens = 1,
            Speculation = new SpeculationOptions
            {
                Enabled = specEnabled,
                SpeculatorName = speculator ?? SpeculatorRegistry.Auto,
                MaxDraftTokens = 8,
                MinDraftProb = speculator == SpeculatorRegistry.NGram ? null : 0.5f,
            },
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var greedy = new SamplingConfig
        {
            Temperature = 0f,
            TopK = 0,
            TopP = 1f,
            RepetitionPenalty = 1f,
            PresencePenalty = 0f,
            FrequencyPenalty = 0f,
        };
        var seq = new SequenceState("mtp-req", Enumerable.Range(1, promptLen).ToList(),
            maxNewTokens, BlockSize, greedy);
        var handle = engine.SubmitRequest(seq);
        handle.Completion.GetAwaiter().GetResult();
        beforeDispose?.Invoke();
        return seq;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    // ----- sampled verification (SpeculativeDecoder.GenerateSampled) -----

    private static SamplingConfig SampledChatConfig(float repeatPenalty = 1f) => new()
    {
        Temperature = 0.8f,
        TopK = 40,
        TopP = 0.95f,
        MinP = 0.05f,
        RepetitionPenalty = repeatPenalty,
        PenaltyLastN = 64,
        Seed = 4242,
    };

    [Theory]
    [InlineData(1f)]
    [InlineData(1.15f)]
    public void GenerateSampled_WithAcceptedDrafts_EmitsWhatPlainSamplingWouldHave(float repeatPenalty)
    {
        // The reference is the SAME loop with drafting gated off, so the two runs
        // differ in exactly one thing: whether tokens arrived through an accepted
        // draft window. That is what the CLI's --mtp-spec buys, and what must not
        // change the stream.
        //
        // On this fake the drafts are correct by construction, so acceptance is
        // near-total and the accept path — onDraftAccepted feeding the penalty
        // history before the next row is drawn, and the drawn token being emitted
        // as drawn rather than re-drawn — is exercised on nearly every token. The
        // synthetic-GLM suite covers the opposite regime (drafts almost always
        // rejected) against real model arithmetic.
        const int promptLen = 5;
        const int maxNew = 24;
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();

        var plainModel = new FakeSpeculativeModel();
        var plainDecoder = new SpeculativeDecoder(plainModel, maxDraftTokens: 4)
        {
            AdaptiveSpeculation = false,
            // Above 1: no confidence can clear it, so every step degrades to a
            // plain single-token decode through the identical code path.
            MinDraftProb = 2f,
        };
        List<int> plain = plainDecoder.GenerateSampled(
            prompt, maxNew, new TokenSampler(SampledChatConfig(repeatPenalty)));
        Assert.Equal(0, plainDecoder.TokensDrafted);
        Assert.Equal(maxNew, plainDecoder.PlainSteps + 1);

        var specModel = new FakeSpeculativeModel();
        var specDecoder = new SpeculativeDecoder(specModel, maxDraftTokens: 4)
        {
            AdaptiveSpeculation = false,
        };
        List<int> speculative = specDecoder.GenerateSampled(
            prompt, maxNew, new TokenSampler(SampledChatConfig(repeatPenalty)));

        Assert.Equal(maxNew, speculative.Count);
        Assert.Equal(plain, speculative);
        Assert.True(specDecoder.TokensAccepted > 0,
            $"no draft was accepted (drafted={specDecoder.TokensDrafted}), so the accept path never ran");
        Assert.True(specDecoder.VerifySteps < plainDecoder.PlainSteps,
            "speculation took no fewer trunk passes than plain decoding");
        Assert.Empty(specModel.ProtocolViolations);
    }

    [Fact]
    public void CarryPosition_AfterAFullRun_AllowsAContinuationToExtendTheCache()
    {
        // A chat session extends its KV cache across turns by prefilling only the
        // new suffix. That is sound exactly when the draft head's carry-in hidden
        // state still describes the token the trunk ends on.
        const int promptLen = 5;
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();

        var model = new FakeSpeculativeModel();
        var decoder = new SpeculativeDecoder(model, maxDraftTokens: 4) { AdaptiveSpeculation = false };
        Assert.Equal(-1, decoder.CarryPosition);

        List<int> produced = decoder.GenerateGreedy(prompt, 12);

        Assert.Equal(12, produced.Count);
        Assert.Equal(model.CacheSeqLen, decoder.CarryPosition);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void CarryPosition_AfterARunStoppedMidWindow_RefusesToResume()
    {
        // A consumer that stops part-way through a verified window (a stop
        // sequence, Ctrl+C, an EOS the draft head predicted early) leaves the
        // trunk holding rows that never reached the caller. Those get rewound —
        // and the rewind drops the row the carry-in hidden state was taken from.
        //
        // Resuming there is silent: the emitted stream would stay correct because
        // verification is trunk-driven, but the draft head picks up one KV row
        // built from the wrong hidden state and acceptance decays from there with
        // nothing in the log to explain it. So the decoder has to say it cannot
        // resume, and the caller re-prefills instead.
        const int promptLen = 5;
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();

        var model = new FakeSpeculativeModel();
        var decoder = new SpeculativeDecoder(model, maxDraftTokens: 4) { AdaptiveSpeculation = false };

        // Drafts are correct by construction here, so the first speculative step
        // accepts its whole 4-token window and commits 5 tokens to the trunk.
        // Stopping after 3 emitted tokens (the prompt's own first token plus two
        // of that window) therefore strands two committed rows.
        int emitted = 0;
        List<int> produced = decoder.GenerateGreedy(prompt, 12, onToken: _ => ++emitted < 3);

        Assert.Equal(3, produced.Count);
        Assert.Equal(-1, decoder.CarryPosition);
        // The trunk is left holding exactly what was returned, so a caller
        // mirroring the cache does not have to reason about where it stopped.
        Assert.Equal(promptLen + produced.Count, model.CacheSeqLen);
        Assert.Empty(model.ProtocolViolations);
    }

    /// <summary>
    /// Deterministic NextN/MTP fake. The trunk's argmax for the row of the
    /// token at absolute position p is <see cref="ExpectedNext"/>(p); the
    /// draft head predicts the same value (peaked, high confidence) except at
    /// <see cref="DraftWrongPositions"/> (off-by-one prediction) and
    /// <see cref="LowConfidencePositions"/> (flat logits, fails the p_min
    /// gate). Hidden-state rows encode <c>position + 1</c> so the fake can
    /// assert the (token, previous-hidden) pairing invariant that drafting
    /// and MTP catch-up depend on; violations are recorded in
    /// <see cref="ProtocolViolations"/> rather than thrown so a buggy caller
    /// fails the test with a readable message instead of a hung engine.
    /// </summary>
    // ----- block drafting (DSpark) -----

    [Fact]
    public void BlockDraft_PerfectBlock_AcceptsWholeBlockInOneDraftCall()
    {
        var model = new FakeSpeculativeModel { BlockDraftSize = 5 };
        var exec = NewExec(model, maxDraftTokens: 8);

        // The window is clamped to the drafter's trained block size.
        Assert.Equal(5, exec.MaxDraftTokens);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        var accepted = new List<int>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 8,
            drawNext: Argmax, onDraftAccepted: accepted.Add);

        Assert.True(outcome.UsedSpeculation);
        Assert.Equal(5, outcome.AcceptedCount);
        Assert.Equal(1, model.BlockDraftCalls); // ONE pass for the whole block
        Assert.Equal(model.ExpectedNext(pos + 5), outcome.NextToken);
        Assert.Equal(pos + 6, model.CacheSeqLen);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void BlockDraft_CumulativeConfidenceGate_TruncatesOnPrefixProbability()
    {
        // Every position is individually above the gate, but the prefix product
        // falls below it at position 3 (0.8^3 = 0.512 >= 0.5 > 0.8^4 = 0.41).
        var model = new FakeSpeculativeModel
        {
            BlockDraftSize = 5,
            BlockConfidences = new[] { 0.8f, 0.8f, 0.8f, 0.8f, 0.8f },
        };
        var exec = NewExec(model, maxDraftTokens: 5, minDraftProb: 0.5f);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        var accepted = new List<int>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 5,
            drawNext: Argmax, onDraftAccepted: accepted.Add);

        Assert.Equal(3, exec.Stats.TokensDrafted);
        Assert.Equal(3, outcome.AcceptedCount);
        Assert.Equal(pos + 4, model.CacheSeqLen);
    }

    [Fact]
    public void BlockDraft_WrongDraftMidBlock_KeepsPrefixAndCorrects()
    {
        var model = new FakeSpeculativeModel { BlockDraftSize = 5 };
        var exec = NewExec(model, maxDraftTokens: 5);

        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        model.DraftWrongPositions.Add(pos + 2); // third drafted token is wrong

        var accepted = new List<int>();
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 5,
            drawNext: Argmax, onDraftAccepted: accepted.Add);

        Assert.Equal(2, outcome.AcceptedCount);
        Assert.Equal(model.ExpectedNext(pos + 2), outcome.NextToken);
        Assert.Equal(pos + 3, model.CacheSeqLen);
        Assert.Equal(1, exec.Stats.RollbackSteps);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void PrefillSelfCatchUp_SkipsPerRowCaptureAndCatchUp()
    {
        var model = new FakeSpeculativeModel { BlockDraftSize = 5, PrefillSelfCatchUp = true };
        var exec = NewExec(model, maxDraftTokens: 5);

        var prompt = new int[6];
        for (int i = 0; i < prompt.Length; i++)
            prompt[i] = i + 1;
        exec.PrefillStep(prompt, 0);

        // The drafter still gets the last prompt row's hidden state, and the
        // block draft's protocol check would flag it if it did not.
        var outcome = exec.DecodeStep(model.ExpectedNext(prompt.Length - 1), prompt.Length, kMax: 5,
            drawNext: Argmax);
        Assert.True(outcome.UsedSpeculation);
        Assert.Empty(model.ProtocolViolations);
    }

    // ----- the confidence gate defaults to the drafter's kind -----

    [Fact]
    public void MinDraftProb_DefaultsToTheGateMatchingTheDrafterKind()
    {
        // The two gates threshold different quantities — a per-token head's
        // top-1 probability vs a block's CUMULATIVE prefix probability — so one
        // shared default badly mis-gates whichever it wasn't chosen for.
        var perToken = NewExec(new FakeSpeculativeModel(), maxDraftTokens: 4);
        Assert.Equal(DraftHeadSpeculator.DefaultGate, perToken.MinDraftProb);

        var block = NewExec(new FakeSpeculativeModel { BlockDraftSize = 5 }, maxDraftTokens: 8);
        Assert.Equal(BlockDraftSpeculator.DefaultGate, block.MinDraftProb);
        // The two are not on a common scale (single top-1 probability vs a
        // cumulative product), so only their identity is asserted, not an order.
        Assert.NotEqual(block.MinDraftProb, perToken.MinDraftProb);
        // The window still clamps to what the drafter was trained to propose.
        Assert.Equal(5, block.MaxDraftTokens);
    }

    [Fact]
    public void MinDraftProb_BlockDefault_KeepsRealisticBlocksInsteadOfDroppingThem()
    {
        // Regression for the server default: a per-token-sized 0.75 gate against
        // the cumulative product truncated nearly every block to nothing, so
        // speculation cost the drafter's time and bought no tokens.
        var model = new FakeSpeculativeModel
        {
            BlockDraftSize = 5,
            BlockConfidences = new[] { 0.9f, 0.85f, 0.8f, 0.75f, 0.7f },
        };
        var exec = NewExec(model, maxDraftTokens: 5);
        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        var outcome = exec.DecodeStep(lastToken, pos, kMax: 5, drawNext: Argmax);

        // 0.9*0.85 = 0.765, *0.8 = 0.612, *0.75 = 0.459, *0.7 = 0.321 -> 4 kept.
        Assert.Equal(4, exec.Stats.TokensDrafted);
        Assert.True(outcome.UsedSpeculation);

        // The same block under a per-token-SIZED gate keeps half as many: the
        // product falls under 0.75 already at the third position (0.612).
        var strictModel = new FakeSpeculativeModel { BlockDraftSize = 5, BlockConfidences = model.BlockConfidences };
        var strict = NewExec(strictModel, maxDraftTokens: 5,
            minDraftProb: 0.75f);   // a per-token-SIZED gate, not today's default
        int pos2 = PrefillPrompt(strictModel, strict, promptLen: 5, out int lastToken2);
        strict.DecodeStep(lastToken2, pos2, kMax: 5, drawNext: Argmax);
        Assert.Equal(2, strict.Stats.TokensDrafted);
    }

    // ----- streaming / resumable generation (the interactive chat path) -----

    [Fact]
    public void GenerateGreedy_StreamsEveryEmittedTokenAndWithholdsTheStopToken()
    {
        var model = new FakeSpeculativeModel { BlockDraftSize = 5, PrefillSelfCatchUp = true };
        var decoder = new SpeculativeDecoder(model, maxDraftTokens: 5);

        var prompt = new[] { 1, 2, 3, 4, 5 };
        // Generated token j is the trunk's prediction at position prompt.Length-1+j,
        // so this ends the run on the 8th one.
        int eos = model.ExpectedNext(prompt.Length + 6);

        var streamed = new List<int>();
        var output = decoder.GenerateGreedy(prompt, maxNewTokens: 64,
            isStopToken: t => t == eos,
            onToken: t => { streamed.Add(t); return true; });

        Assert.Equal(eos, output[^1]);
        // Everything except the stop token reaches the consumer, in order.
        Assert.Equal(output.Take(output.Count - 1), streamed);
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void GenerateGreedy_ConsumerStopsMidBlock_TrunkStaysAPrefixOfTheOutput()
    {
        // The invariant the chat session's KV bookkeeping rests on: on return the
        // trunk holds exactly a PREFIX OF THE RETURNED OUTPUT. A block step commits
        // its whole accepted window in one forward, so stopping part-way through
        // emitting it runs the trunk past the result — the decoder has to drop that
        // tail rather than leave the caller's cache mirror guessing.
        var model = new FakeSpeculativeModel { BlockDraftSize = 5, PrefillSelfCatchUp = true };
        var decoder = new SpeculativeDecoder(model, maxDraftTokens: 5);

        var prompt = new[] { 1, 2, 3, 4, 5 };
        var streamed = new List<int>();
        var output = decoder.GenerateGreedy(prompt, maxNewTokens: 64,
            isStopToken: null,
            onToken: t => { streamed.Add(t); return streamed.Count < 3; });

        Assert.Equal(3, streamed.Count);
        Assert.Equal(streamed, output);
        int committed = model.CacheSeqLen - prompt.Length;
        Assert.True(committed <= output.Count);
        Assert.Equal(output.Take(committed), model.Trunk.Skip(prompt.Length));
    }

    [Fact]
    public void GenerateGreedyFrom_ResumesOnTheCachedPrefixWithoutResettingTheTrunk()
    {
        // The multi-turn chat path: turn 2 prefills only its new tokens on top of
        // the cache turn 1 left behind, then decodes from there.
        var model = new FakeSpeculativeModel { BlockDraftSize = 5, PrefillSelfCatchUp = true };
        var decoder = new SpeculativeDecoder(model, maxDraftTokens: 5);

        var prompt = new[] { 1, 2, 3, 4, 5 };
        var turn1 = decoder.GenerateGreedy(prompt, maxNewTokens: 6);
        int afterTurn1 = model.CacheSeqLen;
        // The trunk holds a prefix of what was generated (the final token is only
        // forwarded when the next step consumes it).
        Assert.InRange(afterTurn1, prompt.Length + turn1.Count - 1, prompt.Length + turn1.Count);

        // Extend the cache with the next turn's tokens only — no ResetKVCache.
        var suffix = new[] { 11, 12, 13 };
        float[] logits = decoder.Prefill(suffix);
        Assert.Equal(afterTurn1 + suffix.Length, model.CacheSeqLen);

        int resumePos = model.CacheSeqLen;
        var turn2 = decoder.GenerateGreedyFrom(logits, resumePos, maxNewTokens: 6);

        Assert.Equal(6, turn2.Count);
        Assert.Equal(model.ExpectedNext(resumePos - 1), turn2[0]);
        Assert.InRange(model.CacheSeqLen, resumePos + turn2.Count - 1, resumePos + turn2.Count);
        // The drafter was fed the right hidden state across the turn boundary.
        Assert.Empty(model.ProtocolViolations);
        Assert.True(decoder.TokensDrafted > 0);
    }

    private sealed class FakeSpeculativeModel : IBatchedSpeculativeModel
    {
        private readonly List<int> _trunk = new();
        private int _recurrentState;       // advances with the trunk
        private int _recurrentSnapshot = -1;

        // Block drafting (DSpark): a positive size routes the executor through
        // DraftBlock, and BlockConfidences is what the confidence head
        // reports for each block position.
        public int BlockDraftSize { get; set; }
        public float[] BlockConfidences { get; set; }
        public int BlockDraftCalls { get; private set; }
        public bool PrefillSelfCatchUp { get; set; }
        public HashSet<int> DraftWrongPositions { get; } = new();
        public HashSet<int> LowConfidencePositions { get; } = new();
        public List<string> ProtocolViolations { get; } = new();
        public int SnapshotCalls { get; private set; }
        public int RestoreCalls { get; private set; }
        public int EosTokenId { get; set; } = -1;

        // Batched-trunk bookkeeping: when enabled, the engine must serve
        // every speculative trunk pass through SpecForwardBatched (the
        // linear SpecForward must stay untouched).
        public bool BatchedTrunkEnabled { get; set; }
        public int LinearSpecForwardCalls { get; private set; }
        public int BatchedSpecForwardCalls { get; private set; }
        public int SlotSnapshotCalls { get; private set; }
        public int SlotRestoreCalls { get; private set; }

        public FakeSpeculativeModel()
        {
            Tokenizer = new FakeTokenizer(this);
        }

        public void Reset()
        {
            _trunk.Clear();
            _recurrentState = 0;
            _recurrentSnapshot = -1;
            DraftWrongPositions.Clear();
            LowConfidencePositions.Clear();
            ProtocolViolations.Clear();
            SnapshotCalls = RestoreCalls = 0;
            LinearSpecForwardCalls = BatchedSpecForwardCalls = 0;
            SlotSnapshotCalls = SlotRestoreCalls = 0;
            BlockDraftCalls = 0;
        }

        public int ExpectedNext(int position) => ((position * 13) + 7) % VocabSize;

        /// <summary>Tokens the trunk has actually committed to its KV cache.</summary>
        public IReadOnlyList<int> Trunk => _trunk;

        // ---- IModelArchitecture ----
        public ModelConfig Config { get; } = new ModelConfig
        {
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
        };
        public ITokenizer Tokenizer { get; }
        /// <summary>A media turn's injector (see <see cref="FakeInjector"/>); null for text.</summary>
        public IMultimodalInjector Injector { get; set; }
        public IMultimodalInjector MultimodalInjector => Injector;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "fp-mtp-fake";
        public long ComputeKVBlockByteSize(int tokenCount) => tokenCount * 4L;
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src)
        {
            // The MTP engine tests never reuse prefixes, but keep the trunk
            // counter consistent if the executor ever injects.
            while (_trunk.Count < s + n) _trunk.Add(0);
            return true;
        }
        public void TruncateKVCache(int n) { }
        public void ResetKVCache()
        {
            _trunk.Clear();
            _recurrentState = 0;
        }
        public void Dispose() { }

        public int ForwardCalls { get; private set; }
        /// <summary>Whether a hidden-free speculator's plain step may run through
        /// <see cref="Forward"/> (see <see cref="ISpeculativeTarget.SpecPlainStepUsesForward"/>).</summary>
        public bool PlainStepUsesForward { get; set; }
        public bool SpecPlainStepUsesForward => PlainStepUsesForward;
        /// <summary>Whether the fake's per-token head keeps no per-position state
        /// (see <see cref="IDraftHead.DraftHeadResumesAfterGap"/>).</summary>
        public bool ResumesAfterGap { get; set; }
        public bool DraftHeadResumesAfterGap => ResumesAfterGap;

        public float[] Forward(int[] tokens)
        {
            ForwardCalls++;
            // Plain path (MTP disabled / disarmed): same truth stream.
            var logits = new float[VocabSize];
            int startPos = _trunk.Count;
            _trunk.AddRange(tokens);
            _recurrentState = _trunk.Count;
            logits[ExpectedNext(startPos + tokens.Length - 1)] = 10f;
            return logits;
        }

        // ---- ISpeculativeModel ----
        public bool HasDraftHead => true;

        public DraftHeadKind DraftHeadKind =>
            BlockDraftSize > 0 ? DraftHeadKind.Block : DraftHeadKind.PerToken;
        // Mirrors a backend that lacks the accelerated MTP kernels (e.g. the
        // pure-C# CUDA backend for Gemma 4): the engine must NOT engage spec.
        public bool SpeculationProfitable { get; set; } = true;
        public int CacheSeqLen => _trunk.Count;
        public int MaxContextLength => 4096;

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            LinearSpecForwardCalls++;
            ForwardCore(tokens, hAllOut, logitsOut, allLogitsRows);
        }

        private void ForwardCore(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            int startPos = _trunk.Count;
            _trunk.AddRange(tokens);
            _recurrentState = _trunk.Count;

            // A hidden buffer too small for every row means the caller only
            // wants the LAST row (the self-catch-up prefill contract).
            bool lastRowOnly = hAllOut != null && hAllOut.Length < tokens.Length * HiddenSize;

            for (int i = 0; i < tokens.Length; i++)
            {
                if (hAllOut != null && (!lastRowOnly || i == tokens.Length - 1))
                {
                    int row = lastRowOnly ? 0 : i;
                    for (int hh = 0; hh < HiddenSize; hh++)
                        hAllOut[row * HiddenSize + hh] = startPos + i + 1;
                }
                if (allLogitsRows)
                {
                    int rowBase = i * VocabSize;
                    Array.Clear(logitsOut, rowBase, VocabSize);
                    logitsOut[rowBase + ExpectedNext(startPos + i)] = 10f;
                }
            }
            if (!allLogitsRows)
            {
                Array.Clear(logitsOut, 0, VocabSize);
                logitsOut[ExpectedNext(startPos + tokens.Length - 1)] = 10f;
            }
        }

        public void DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            // The hidden chained into a draft at position pos must be the
            // hidden of the token at pos-1 (encoded as pos; zeros before the
            // first prompt token).
            float expectH = pos == 0 ? 0f : pos;
            if (Math.Abs(hPrev[0] - expectH) > 0.001f)
                ProtocolViolations.Add($"draft at pos {pos} got hPrev {hPrev[0]} (expected {expectH})");

            Array.Clear(logitsOut, 0, VocabSize);
            if (!LowConfidencePositions.Contains(pos))
            {
                int predicted = ExpectedNext(pos);
                if (DraftWrongPositions.Contains(pos))
                    predicted = (predicted + 1) % VocabSize;
                logitsOut[predicted] = 10f;
            }
            for (int hh = 0; hh < HiddenSize; hh++)
                hOut[hh] = pos + 1;
        }

        public int DraftBlockSize => BlockDraftSize;

        public bool DraftSelfCatchUp => PrefillSelfCatchUp;

        /// <summary>One-pass block draft: position i predicts the token after
        /// position pos+i, mirroring what a DSpark block does.</summary>
        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            BlockDraftCalls++;
            float expectH = position == 0 ? 0f : position;
            if (Math.Abs(hPrev[0] - expectH) > 0.001f)
                ProtocolViolations.Add($"block draft at pos {position} got hPrev {hPrev[0]} (expected {expectH})");

            int n = Math.Min(BlockDraftSize, draftOut.Length);
            for (int i = 0; i < n; i++)
            {
                int pos = position + i;
                int predicted = ExpectedNext(pos);
                if (DraftWrongPositions.Contains(pos))
                    predicted = (predicted + 1) % VocabSize;
                draftOut[i] = predicted;
                confOut[i] = BlockConfidences != null && i < BlockConfidences.Length ? BlockConfidences[i] : 1.0f;
            }
            return n;
        }

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
            if (hRows == null)
            {
                // A seeded start (SeedCommitted) hands over the committed tokens with
                // no hidden rows; only a head that keeps no per-position state may
                // accept that, and it has nothing to replay.
                if (!ResumesAfterGap)
                    ProtocolViolations.Add($"catch-up at startPos {startPos} without hidden rows on a head that keeps state");
                return;
            }
            for (int k = 0; k < tokens.Length; k++)
            {
                float expectH = startPos + k == 0 ? 0f : startPos + k;
                if (Math.Abs(hRows[k * HiddenSize] - expectH) > 0.001f)
                {
                    ProtocolViolations.Add(
                        $"catch-up row {k} at startPos {startPos} got h {hRows[k * HiddenSize]} (expected {expectH})");
                }
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen) { }

        public void SpecSnapshotRecurrentState()
        {
            SnapshotCalls++;
            _recurrentSnapshot = _recurrentState;
        }

        public void SpecRestoreRecurrentState()
        {
            RestoreCalls++;
            if (_recurrentSnapshot < 0)
                ProtocolViolations.Add("restore without snapshot");
            _recurrentState = _recurrentSnapshot;
        }

        public void SpecRewindCache(int length)
        {
            if (length < 0 || length > _trunk.Count)
            {
                ProtocolViolations.Add($"rewind to {length} outside [0, {_trunk.Count}]");
                return;
            }
            _trunk.RemoveRange(length, _trunk.Count - length);
        }

        // ---- IBatchedSpeculativeModel ----
        public bool SupportsBatchedSpecTrunk => BatchedTrunkEnabled;

        public void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            BatchedSpecForwardCalls++;
            if (startPos != _trunk.Count)
                ProtocolViolations.Add($"batched forward at startPos {startPos} but trunk holds {_trunk.Count}");
            if (seq != null && startPos != seq.NumComputedTokens)
                ProtocolViolations.Add($"batched forward at startPos {startPos} but sequence has {seq.NumComputedTokens} computed");
            if (seq != null && seq.BlockTable.CapacityTokens < startPos + tokens.Length)
                ProtocolViolations.Add($"block table covers {seq.BlockTable.CapacityTokens} tokens but pass needs {startPos + tokens.Length}");
            ForwardCore(tokens, hAllOut, logitsOut, allLogitsRows);
        }

        public void SpecSnapshotRecurrentStateSlots(SequenceState seq)
        {
            SlotSnapshotCalls++;
            _recurrentSnapshot = _recurrentState;
        }

        public void SpecRestoreRecurrentStateSlots(SequenceState seq)
        {
            SlotRestoreCalls++;
            if (_recurrentSnapshot < 0)
                ProtocolViolations.Add("slot restore without snapshot");
            _recurrentState = _recurrentSnapshot;
            // The real batched rollback never rewinds attention KV (reads are
            // extent-bounded and rejected slots get overwritten); model that
            // by truncating to the snapshot position so the next pass's
            // position checks see the rolled-back extent.
            if (_recurrentSnapshot >= 0 && _recurrentSnapshot <= _trunk.Count)
                _trunk.RemoveRange(_recurrentSnapshot, _trunk.Count - _recurrentSnapshot);
        }

        private sealed class FakeTokenizer : ITokenizer
        {
            private readonly FakeSpeculativeModel _owner;
            public FakeTokenizer(FakeSpeculativeModel owner)
            {
                _owner = owner;
                Vocab = new string[SpeculativeExecutionTests.VocabSize];
                for (int i = 0; i < Vocab.Length; i++) Vocab[i] = i.ToString();
            }
            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => _owner.EosTokenId >= 0 ? new[] { _owner.EosTokenId } : Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                    buffer.Add(b);
            }
            public bool IsEos(int tokenId) => _owner.EosTokenId >= 0 && tokenId == _owner.EosTokenId;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
