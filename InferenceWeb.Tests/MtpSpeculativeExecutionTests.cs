// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Unit tests for NextN/MTP speculative decoding: the shared
// MtpSpeculativeExecution core (draft / verify / rollback / catch-up protocol
// against a deterministic fake model — no GGUF needed) and the engine path
// (BatchExecutor routing, ExtraTokens streaming, block-boundary draft capping,
// EOS inside an accepted draft window).
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public class MtpSpeculativeExecutionTests
{
    private const int VocabSize = 64;
    private const int HiddenSize = 4;
    private const int BlockSize = 8;

    // ----- shared-core tests (drive MtpSpeculativeExecution directly) -----

    [Fact]
    public void DecodeStep_PerfectDrafts_AcceptsWholeWindowAndReturnsBonus()
    {
        var model = new FakeMtpModel();
        var exec = new MtpSpeculativeExecution(model, maxDraftTokens: 4);

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
        var model = new FakeMtpModel();
        var exec = new MtpSpeculativeExecution(model, maxDraftTokens: 4);

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
        var model = new FakeMtpModel();
        var exec = new MtpSpeculativeExecution(model, maxDraftTokens: 4);

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
        var model = new FakeMtpModel();
        var exec = new MtpSpeculativeExecution(model, maxDraftTokens: 3);

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
        exec = new MtpSpeculativeExecution(model, maxDraftTokens: 3);
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
    public void EngineMtpSpec_GreedyStream_MatchesPlainDecodeAcrossBlockBoundaries()
    {
        // BlockSize 8 with 40 generated tokens crosses several block
        // boundaries; the executor must cap each draft window to the blocks
        // the scheduler reserved (boundary steps degrade to plain decode)
        // instead of overrunning the block table. A couple of wrong drafts
        // exercise rollback inside the engine flow. The output stream must
        // still equal the model's deterministic plain-greedy chain exactly.
        const int promptLen = 5;
        const int maxNew = 40;

        var model = new FakeMtpModel();
        model.DraftWrongPositions.Add(promptLen + 7);
        model.DraftWrongPositions.Add(promptLen + 19);

        var seq = RunEngineRequest(model, promptLen, maxNew, mtpEnabled: true);

        Assert.Equal(SequenceStatus.FinishedLengthCapped, seq.Status);
        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.NotNull(seq.SpecStats);
        Assert.True(seq.SpecStats.TokensAccepted > 0, "speculation never accepted a draft");
        Assert.True(seq.SpecStats.RollbackSteps >= 1, "wrong drafts should have caused a rollback");
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void EngineMtpSpec_EosInsideAcceptedWindow_StopsAndTruncatesTrailingDrafts()
    {
        const int promptLen = 5;
        var model = new FakeMtpModel();
        // The 6th generated token is EOS; with an 8-token draft window it
        // lands inside an accepted speculative batch.
        int eosToken = model.ExpectedNext(promptLen + 4);
        model.EosTokenId = eosToken;

        var seq = RunEngineRequest(model, promptLen, maxNewTokens: 32, mtpEnabled: true);

        Assert.Equal(SequenceStatus.FinishedStopped, seq.Status);
        // Output ends at EOS (kept, per the engine's contract) with the
        // speculative acceptances past it truncated.
        Assert.Equal(ExpectedChain(model, promptLen, 5), seq.OutputTokens.Take(5));
        Assert.Equal(eosToken, seq.OutputTokens[^1]);
        Assert.Equal(6, seq.OutputTokens.Count);
        Assert.Equal(0, model.ProtocolViolations.Count);
    }

    [Fact]
    public void EngineMtpSpec_Disabled_ProducesSameStreamWithoutSpeculation()
    {
        const int promptLen = 5;
        const int maxNew = 16;
        var model = new FakeMtpModel();

        var seq = RunEngineRequest(model, promptLen, maxNew, mtpEnabled: false);

        Assert.Equal(ExpectedChain(model, promptLen, maxNew), seq.OutputTokens);
        Assert.Null(seq.SpecStats);
    }

    // ----- helpers -----

    private static int PrefillPrompt(FakeMtpModel model, MtpSpeculativeExecution exec, int promptLen, out int lastToken)
    {
        int[] prompt = Enumerable.Range(1, promptLen).ToArray();
        float[] logits = exec.PrefillStep(prompt);
        lastToken = Argmax(logits);
        Assert.Equal(model.ExpectedNext(promptLen - 1), lastToken);
        return promptLen;
    }

    /// <summary>The deterministic greedy stream: token i of the generation is
    /// the trunk's argmax for the row at absolute position promptLen-1+i.</summary>
    private static List<int> ExpectedChain(FakeMtpModel model, int promptLen, int count)
    {
        var expected = new List<int>(count);
        for (int i = 0; i < count; i++)
            expected.Add(model.ExpectedNext(promptLen - 1 + i));
        return expected;
    }

    private static SequenceState RunEngineRequest(FakeMtpModel model, int promptLen, int maxNewTokens, bool mtpEnabled)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 64,
            NumBlocks = 32,
            BlockSize = BlockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
            MtpSpeculativeEnabled = mtpEnabled,
            MtpMaxDraftTokens = 8,
            MtpMinDraftProb = 0.5f,
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
        return seq;
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
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
    private sealed class FakeMtpModel : IMtpSpeculativeModel
    {
        private readonly List<int> _trunk = new();
        private int _recurrentState;       // advances with the trunk
        private int _recurrentSnapshot = -1;

        public HashSet<int> DraftWrongPositions { get; } = new();
        public HashSet<int> LowConfidencePositions { get; } = new();
        public List<string> ProtocolViolations { get; } = new();
        public int SnapshotCalls { get; private set; }
        public int RestoreCalls { get; private set; }
        public int EosTokenId { get; set; } = -1;

        public FakeMtpModel()
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
        }

        public int ExpectedNext(int position) => ((position * 13) + 7) % VocabSize;

        // ---- IModelArchitecture ----
        public ModelConfig Config { get; } = new ModelConfig
        {
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
        };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
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

        public float[] Forward(int[] tokens)
        {
            // Plain path (MTP disabled / disarmed): same truth stream.
            var logits = new float[VocabSize];
            int startPos = _trunk.Count;
            _trunk.AddRange(tokens);
            _recurrentState = _trunk.Count;
            logits[ExpectedNext(startPos + tokens.Length - 1)] = 10f;
            return logits;
        }

        // ---- IMtpSpeculativeModel ----
        public bool HasMtp => true;
        public int CacheSeqLen => _trunk.Count;
        public int MaxContextLength => 4096;

        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            int startPos = _trunk.Count;
            _trunk.AddRange(tokens);
            _recurrentState = _trunk.Count;

            for (int i = 0; i < tokens.Length; i++)
            {
                if (hAllOut != null)
                {
                    for (int hh = 0; hh < HiddenSize; hh++)
                        hAllOut[i * HiddenSize + hh] = startPos + i + 1;
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

        public void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
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

        public void MtpCatchUp(int[] tokens, float[] hRows, int startPos)
        {
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

        public void MtpEnsureCapacity(int requiredSeqLen) { }

        public void MtpSnapshotRecurrentState()
        {
            SnapshotCalls++;
            _recurrentSnapshot = _recurrentState;
        }

        public void MtpRestoreRecurrentState()
        {
            RestoreCalls++;
            if (_recurrentSnapshot < 0)
                ProtocolViolations.Add("restore without snapshot");
            _recurrentState = _recurrentSnapshot;
        }

        public void MtpRewindCache(int length)
        {
            if (length < 0 || length > _trunk.Count)
            {
                ProtocolViolations.Add($"rewind to {length} outside [0, {_trunk.Count}]");
                return;
            }
            _trunk.RemoveRange(length, _trunk.Count - length);
        }

        private sealed class FakeTokenizer : ITokenizer
        {
            private readonly FakeMtpModel _owner;
            public FakeTokenizer(FakeMtpModel owner)
            {
                _owner = owner;
                Vocab = new string[MtpSpeculativeExecutionTests.VocabSize];
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
