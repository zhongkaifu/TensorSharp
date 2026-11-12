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
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// The shared draft/verify core - the ONE place speculative decoding is
    /// implemented, for every algorithm and every model. Driven either by the
    /// engine's <c>BatchExecutor</c> (sampler-verified, one step per scheduler
    /// iteration) or by the standalone <c>SpeculativeDecoder</c> in
    /// TensorSharp.Models (which owns its own generate loop).
    ///
    /// Per step:
    ///   1. DRAFT: <see cref="ISpeculator.Propose"/> proposes up to
    ///      <see cref="MaxDraftTokens"/> tokens. This class does not know or
    ///      care how - a chained NextN head, a one-pass block drafter, an
    ///      n-gram lookup over the context, or something not written yet.
    ///   2. VERIFY: the trunk forwards [lastToken, d1..dK] as ONE batch with
    ///      per-row logits; the caller's <c>drawNext</c> draws each row with the
    ///      request's own sampler and drafts are accepted while the drawn token
    ///      matches; row m's drawn token is the corrected/bonus token for free.
    ///   3. ROLLBACK: on partial acceptance the recurrent (GDN) state is
    ///      restored from a pre-verify snapshot and re-advanced over the kept
    ///      prefix; attention KV only needs a position rewind. Trunks whose
    ///      verify already persisted usable KV skip the re-forward entirely.
    ///   4. COMMIT: kept tokens are handed back to the speculator with their
    ///      exact trunk hidden states, so a learned drafter's KV cache tracks
    ///      the real context and a lookup drafter's corpus tracks the real
    ///      output.
    ///
    /// WHY THE OUTPUT CANNOT CHANGE: every emitted token is drawn by the
    /// CALLER's sampler from a TRUNK row. The draft only decides whether that
    /// row had already been computed. A wrong draft costs a rollback, never a
    /// wrong token, which is why <see cref="ISpeculator"/> carries no
    /// correctness obligations at all.
    ///
    /// Single-sequence; the caller owns the model's KV cache lifecycle and the
    /// position bookkeeping (a step at <c>position</c> advances the trunk to
    /// <c>position + AcceptedCount + 1</c>).
    /// </summary>
    public sealed class SpeculativeExecution : IDisposable
    {
        private readonly ISpeculativeTarget _model;
        private readonly ISpeculator _speculator;
        private readonly ISpecTrunk _trunk;
        [MemberNotNullWhen(true, nameof(_pendingH), nameof(_verifyH), nameof(_catchUpH))]
        private bool NeedsHidden => _pendingH != null && _verifyH != null && _catchUpH != null;
        private readonly int _hidden;
        private readonly int _vocab;

        // Trunk hidden state of the token immediately BEFORE the next pending
        // token (llama.cpp's pending_h). Zeros before the first prompt token.
        // Null when the speculator needs no hidden state.
        private readonly float[]? _pendingH;

        // Reusable buffers (speculative windows are small; prefill chunk
        // buffers grow to the largest chunk seen).
        private readonly float[] _verifyLogits;  // [(K+1) * vocab]
        private readonly float[]? _verifyH;       // [(K+1) * hidden]
        private readonly float[]? _catchUpH;      // [(K+1) * hidden]
        private readonly float[] _stepLogits;    // [vocab] for plain/re-advance steps
        private readonly float[] _rowLogits;     // [vocab] scratch row handed to drawNext
        private readonly int[] _verifyDrawn;     // optional plain-greedy verify, one first-max id per row
        private readonly int[] _oneToken = new int[1];

        /// <summary>Logits handed back for a plain step; see the allocation note at
        /// its only assignment.</summary>
        private float[]? _plainOut;

        /// <summary>Consecutive decode steps on which the drafter proposed nothing.
        /// Reset by any verify.</summary>
        private int _plainRun;

        /// <summary>
        /// How long a run of plain steps has to get before it is worth changing
        /// graph family for. One switch amortized over this many tokens is a few
        /// percent; staying on the speculative family for an unbounded run is a
        /// multiple. Small, because the drafter re-arms on the very next verify.
        /// </summary>
        private const int PlainRunBeforeSwitch = 2;
        private readonly List<int> _draftTokens = new();
        private float[]? _chunkH;                 // [chunk * hidden] prefill h capture, shifted in place into (token k, h of k-1) pairs
        private float[]? _lastRowH;               // [hidden] the row the in-place shift would otherwise overwrite

        /// <summary>The algorithm doing the drafting. Exposed for logs and for
        /// callers that want to inspect or retune it mid-run.</summary>
        public ISpeculator Speculator => _speculator;

        /// <summary>The cost governor guarding this execution. Set
        /// <c>Enabled = false</c> to force drafting on for A/B measurement.</summary>
        public SpeculationCostGovernor Governor { get; }

        /// <summary>Maximum tokens drafted per step, bounded by the trunk's
        /// verification capacity even when the algorithm was constructed directly.</summary>
        public int MaxDraftTokens { get; }

        /// <summary>The drafter's confidence gate; see
        /// <see cref="ISpeculator.MinDraftProb"/> for what the number means for
        /// the algorithm in use.</summary>
        public float MinDraftProb
        {
            get => _speculator.MinDraftProb;
            set => _speculator.MinDraftProb = value;
        }

        /// <summary>Measure speculation against plain decoding at runtime and
        /// skip drafting while it is measurably slower. On by default.</summary>
        public bool AdaptiveSpeculation
        {
            get => Governor.Enabled;
            set => Governor.Enabled = value;
        }

        /// <summary>True when the trunk keeps the speculator in sync itself, so
        /// prompt prefill needs neither per-row hidden states nor a commit.</summary>
        public bool PrefillSelfCatchUp => _speculator.HandlesOwnPrefill;

        public SpeculationStats Stats { get; } = new();

        // Tag of the process-wide drafted/accepted counters (TensorSharpMetrics), built once per
        // execution so a verify step records them without allocating. The counters add the same
        // deltas as Stats, here and not at completion: a preempted request that re-arms gets a
        // new execution, and work its earlier one did would otherwise go uncounted.
        private readonly KeyValuePair<string, object?> _architectureTag;

        public SpeculativeExecution(ISpeculativeTarget model, ISpeculator speculator, ISpecTrunk? trunk = null,
            SpeculationCostGovernor? governor = null)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _speculator = speculator ?? throw new ArgumentNullException(nameof(speculator));
            _trunk = trunk ?? new LinearSpecTrunk(model);
            // A governor handed in is SHARED: its verdict about this (model, drafter,
            // backend) carries from one request to the next, so a chat's short turns
            // do not each pay a fresh probe round to rediscover that drafting loses
            // on prose (it re-probes with backoff either way).
            // This constructor has no sampler context; its fallback cannot opt
            // into the measured greedy-only default. HTTP supplies that profile.
            Governor = governor ?? SpeculationGovernorFactory.CreateFor(model, speculator);
            Governor.NewSequence();   // a shared governor: this request starts fresh enough to re-probe
            _governorWinsAtStart = Governor.Wins;
            _governorLossesAtStart = Governor.Losses;
            _governorParkedAtStart = Governor.ParkedSteps;

            _hidden = model.SpecFeatureSize;
            _vocab = model.Config.VocabSize;
            string architecture = model.Config.Architecture;
            _architectureTag = new(Telemetry.TelemetryAttributes.ModelArchitecture,
                string.IsNullOrEmpty(architecture) ? "unknown" : architecture);

            int k = Math.Min(speculator.MaxDraftTokens, Math.Max(0, model.SpecMaxDraftTokens));
            MaxDraftTokens = k;
            _verifyLogits = new float[(k + 1) * (long)_vocab];
            _stepLogits = new float[_vocab];
            _rowLogits = new float[_vocab];
            _verifyDrawn = new int[k + 1];
            if (speculator.NeedsHiddenState)
            {
                _pendingH = new float[_hidden];
                _verifyH = new float[(k + 1) * _hidden];
                _catchUpH = new float[(k + 1) * _hidden];
            }
        }

        /// <summary>
        /// Start the statistics and the cost governor over WITHOUT touching the
        /// carry-in hidden state, for a caller that reports per turn while keeping
        /// the drafter aligned with a KV cache it is extending. Resetting the
        /// stats alone would leave the governor's round accumulators running behind
        /// zeroed counters - a per-turn log that reads 0.0 ms/token - and would
        /// carry a park decided on one turn's context into the next.
        /// </summary>
        public void ResetStatsAndGovernor()
        {
            Stats.Reset();
            Governor.Reset();
        }

        /// <summary>
        /// Arm over a context the trunk already holds: hand the speculator the
        /// <paramref name="committedTokens"/> that are in the cache (positions 0..n-1)
        /// so its own state - an n-gram corpus - covers them, without forwarding
        /// anything. For a request whose prefill ran on the plain path (a media turn,
        /// whose image/audio embeddings only that path can inject) and that would
        /// otherwise decode plainly to the end. Only a speculator that needs no
        /// hidden state can start this way: there is no trunk hidden state for the
        /// last committed token to chain a learned head from.
        /// </summary>
        public void SeedCommitted(int[] committedTokens)
        {
            ArgumentNullException.ThrowIfNull(committedTokens);
            if (!CanSeedCommitted)
                throw new InvalidOperationException(
                    $"{_speculator.Describe()} keeps per-position state it never replayed and cannot start mid-sequence.");
            if (committedTokens.Length > 0)
                _speculator.Commit(committedTokens, null, 0);
            // A learned head that resumes after a gap still needs the trunk hidden
            // state of the last committed token, which no plain prefill captured:
            // the first step runs plain and captures it, drafting starts at the second.
            _carryValid = !NeedsHidden;
        }

        /// <summary>
        /// The trunk advanced past this execution without it - a concurrent interlude
        /// served the request plainly on the batched or serial fused path. Hand the
        /// speculator those tokens (no hidden rows: a head that resumes after a gap
        /// accepts that and takes one plain step to recapture its carry) and let the
        /// governor count the steps, so the request resumes where it was instead of
        /// being re-armed from scratch: a re-arm re-indexed the whole context and
        /// re-ran the probe on every 1 -> N -> 1 transition. Requires
        /// <see cref="CanSeedCommitted"/>.
        /// </summary>
        public void CatchUp(int[] tokens, int startPos)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (!CanSeedCommitted)
                throw new InvalidOperationException(
                    $"{_speculator.Describe()} keeps per-position state it never replayed and cannot resume after a gap.");
            if (tokens.Length == 0)
                return;
            _speculator.Commit(tokens, null, startPos);
            _carryValid = !NeedsHidden;
            Governor.NoteExternalPlainSteps(tokens.Length);
        }

        /// <summary>
        /// Whether <see cref="SeedCommitted"/> can arm this execution over tokens the
        /// trunk already holds: a speculator with no hidden-state chain (n-gram), or a
        /// learned head that keeps no per-position state of its own
        /// (<see cref="ISpeculator.CanArmAfterPrefixReuse"/>).
        /// </summary>
        public bool CanSeedCommitted => !NeedsHidden || _speculator.CanArmAfterPrefixReuse;

        /// <summary>
        /// Whether a fresh prompt should prefill through this execution rather than the plain path, even when
        /// <see cref="CanSeedCommitted"/> would arm it afterwards: the speculator's proposals read the prompt's hidden
        /// rows (<see cref="ISpeculator.ConditionsOnPrompt"/>) and the target does not feed them during its own
        /// forward (<see cref="ISpeculator.HandlesOwnPrefill"/>).
        /// </summary>
        public bool PrefillsThroughContext => NeedsHidden && _speculator.ConditionsOnPrompt && !_speculator.HandlesOwnPrefill;

        // False until the trunk hidden state that drafting chains from has been
        // captured by a forward of this execution (see SeedCommitted).
        private bool _carryValid = true;

        // Always known when the model keeps it (ISpeculativeTarget.SpecCarryOnDevice).
        private bool CarryValid => _carryValid || _model.SpecCarryOnDevice;

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset()
        {
            if (_pendingH != null)
                Array.Clear(_pendingH);
            _carryValid = true;   // the prefill that follows a reset produces the carry
            _speculator.Reset();
            Stats.Reset();
            Governor.Reset();
        }

        /// <summary>
        /// Forward one prompt chunk through the trunk (capturing hidden states
        /// where the speculator needs them) and hand the chunk to the
        /// speculator so its own state covers it.
        /// <paramref name="startPos"/> is the chunk's first trunk position
        /// (chunks are contiguous from position 0); the trunk must be
        /// positioned there. Returns a caller-owned copy of the last-position
        /// logits.
        /// </summary>
        public float[] PrefillStep(int[] chunk, int startPos, int promptEnd = -1)
        {
            if (chunk == null || chunk.Length == 0)
                throw new ArgumentException("Chunk must not be empty.", nameof(chunk));

            int n = chunk.Length;

            if (!NeedsHidden)
            {
                _trunk.Forward(chunk, null, _stepLogits, allLogitsRows: false);
                _speculator.Commit(chunk, null, startPos);
            }
            else if (_speculator.HandlesOwnPrefill)
            {
                // The trunk keeps the drafter in sync itself and only hands
                // back the last row's hidden state.
                _trunk.Forward(chunk, _pendingH, _stepLogits, allLogitsRows: false);
            }
            else if (promptEnd > 0 && _speculator.PrefillContextRows < int.MaxValue
                && (long)startPos + n <= (long)promptEnd - _speculator.PrefillContextRows)
            {
                // Every row of this chunk is older than the drafter's context at the
                // end of the prompt, so later rows overwrite whatever it would write.
                // Keep only the last row's hidden state: the next chunk's carry-in.
                // A null-row commit reports the gap.
                EnsureChunkBuffers(1);
                _trunk.Forward(chunk, _lastRowH, _stepLogits, allLogitsRows: false);
                _speculator.Commit(chunk, null, startPos);
                Array.Copy(_lastRowH, 0, _pendingH, 0, _hidden);
            }
            else
            {
                EnsureChunkBuffers(n);

                _trunk.Forward(chunk, _chunkH, _stepLogits, allLogitsRows: false);

                // Pair token k with the hidden state of the token before it. Done
                // IN PLACE: the only row still needed after the shift is the last
                // one (it becomes the next chunk's carry-in), so save that, slide
                // the block down a row and prepend the carried-in row. A second
                // chunk-sized buffer used to hold the result, which for a DFlash
                // feature row (33280 floats, 130 KB) is 130 MB at a 1024-row chunk
                // - and doubling the prefill chunk is worth far more than the
                // buffer it costs. Array.Copy is memmove-safe on overlap.
                Array.Copy(_chunkH, (long)(n - 1) * _hidden, _lastRowH, 0, _hidden);
                if (n > 1)
                    Array.Copy(_chunkH, 0, _chunkH, _hidden, (long)(n - 1) * _hidden);
                Array.Copy(_pendingH, 0, _chunkH, 0, _hidden);
                _speculator.Commit(chunk, _chunkH, startPos);

                Array.Copy(_lastRowH, 0, _pendingH, 0, _hidden);
            }

            float[] logits = new float[_vocab];
            Array.Copy(_stepLogits, logits, _vocab);
            return logits;
        }

        /// <summary>
        /// One speculative decode step for <paramref name="lastToken"/> at trunk
        /// position <paramref name="position"/> (== tokens already in the cache).
        /// <paramref name="kMax"/> additionally caps this step's draft window
        /// (token budget, KV block capacity); the effective window is
        /// min(kMax, <see cref="MaxDraftTokens"/>, context headroom) and a
        /// non-positive window degrades to a plain decode.
        /// <paramref name="drawNext"/> draws a token from a verify-row logits
        /// copy with the caller's sampler; <paramref name="onDraftAccepted"/>
        /// fires for each accepted draft BEFORE the next row is drawn so the
        /// caller can keep its penalty history exact;
        /// <paramref name="adjustDraftLogits"/> (optional) mutates draft logits
        /// in place given the drafts pending in this window;
        /// <paramref name="history"/> (optional) is the caller's emitted-token
        /// history, for algorithms that mine it.
        /// <paramref name="drawGreedy"/> explicitly authorizes token-only verify
        /// for a plain greedy sampler: it receives raw first-max ids and must apply
        /// the same forced-token and stop checks as drawNext. A successful token-only
        /// verify returns null NextLogits; the drawn NextToken must be retained.
        /// The trunk advances to <c>position + AcceptedCount + 1</c>.
        /// </summary>
        public SpeculativeStepOutcome DecodeStep(
            int lastToken,
            int position,
            int kMax,
            Func<float[], int> drawNext,
            Action<float[], IReadOnlyList<int>>? adjustDraftLogits = null,
            Action<int>? onDraftAccepted = null,
            IReadOnlyList<int>? history = null,
            Func<int, int>? drawGreedy = null)
        {
            ArgumentNullException.ThrowIfNull(drawNext);

            long tStep0 = Stopwatch.GetTimestamp();
            // Governor: a step it declines becomes a plain decode, which is the
            // same path an all-low-confidence draft window already takes (and so
            // keeps the drafter's own state in sync via the commit below).
            bool governorDeclined = false;
            if (!Governor.AllowsSpeculation())
            {
                governorDeclined = true;
                kMax = 0;
                // Only a genuine park counts. The calibration plain steps that a
                // round takes to establish its baseline also come through here, and
                // counting them made a perfectly healthy run report "parked 3",
                // which is indistinguishable from a short real park in a log.
                if (Governor.IsParked)
                    Stats.ParkedSteps++;
            }

            kMax = Math.Min(kMax, Math.Min(MaxDraftTokens, _model.SpecSoloMaxDraftTokens));
            // Verify needs position + K + 1 trunk slots; drafting writes drafter
            // rows up to position + K.
            kMax = Math.Min(kMax, _model.MaxContextLength - position - 2);
            // No carried hidden state yet (armed over a plain prefill): this step
            // runs plain and captures it.
            if (!CarryValid)
                kMax = 0;

            _draftTokens.Clear();
            if (kMax > 0)
            {
                long tDraft0 = Stopwatch.GetTimestamp();
                _model.SpecEnsureCapacity(position + kMax + 1);
                _speculator.Propose(
                    new DraftContext
                    {
                        LastToken = lastToken,
                        Position = position,
                        MaxTokens = kMax,
                        CarryHidden = _pendingH,
                        History = history,
                        AdjustLogits = adjustDraftLogits,
                    },
                    _draftTokens);
                if (_draftTokens.Count > kMax)
                    throw new InvalidOperationException(
                        $"The {_speculator.Name} speculator proposed {_draftTokens.Count} tokens "
                        + $"for a verification window limited to {kMax} drafts.");
                Stats.DraftTicks += Stopwatch.GetTimestamp() - tDraft0;
            }

            if (_draftTokens.Count == 0)
            {
                // Plain decode step (still captures h + keeps the drafter in sync).
                Stats.PlainSteps++;
                _plainRun++;
                long tPlain0 = Stopwatch.GetTimestamp();
                _oneToken[0] = lastToken;
                // A seeded start has no carry for this token: hand the speculator the
                // token with no hidden row (a head that starts mid-sequence accepts
                // that, by contract) rather than a row of zeros dressed as one.
                bool carryKnown = CarryValid;
                // While the GOVERNOR has parked a learned head that can resume after
                // a gap, the hidden state is not needed - nothing will draft from it
                // until the park ends - so the step takes the model's own decode when
                // the trunk offers it, and the first step after the park captures the
                // carry again. Otherwise the park was measured against a plain step
                // that is itself dearer than the real one (a one-row speculative
                // forward with a per-op head), and the governor kept calling
                // speculation a win against that inflated baseline: E4B-IQ4_XS on the
                // host benchmark ran prose at 23-40 tok/s under it, against 65 plain.
                bool cheapPlain = !NeedsHidden
                    || (governorDeclined && _speculator.CanArmAfterPrefixReuse && _trunk.HasCheapPlainStep);
                // A model whose plain step costs a graph-family switch keeps
                // isolated plain steps in the speculative family, because paying the
                // switch twice for one token is dearer than the step it saves. That
                // reasoning stops holding once plain steps stop being isolated: a
                // drafter that proposes nothing for a long stretch was leaving every
                // one of those tokens on the dear family (93 of 101 steps on a
                // measured Qwen 3.5 run, 50 ms each against 9.2 ms for the model's
                // own decode). After PlainRunBeforeSwitch consecutive plain steps the
                // switch is amortized by the run itself, so take it and stay there
                // until something actually drafts.
                bool plainRunDominates = _plainRun > PlainRunBeforeSwitch;
                if (cheapPlain)
                    _trunk.ForwardPlain(lastToken, _stepLogits,
                        parked: governorDeclined || plainRunDominates);   // the model's own decode step
                else
                    _trunk.Forward(_oneToken, _verifyH, _stepLogits, allLogitsRows: false);
                if (NeedsHidden)
                    Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                _speculator.Commit(_oneToken, NeedsHidden && !carryKnown ? null : _catchUpH, position);
                if (NeedsHidden)
                {
                    if (cheapPlain)
                    {
                        _carryValid = false;   // captured again by the first step after the park
                    }
                    else
                    {
                        Array.Copy(_verifyH, 0, _pendingH, 0, _hidden);
                        _carryValid = true;
                    }
                }
                Stats.PlainTicks += Stopwatch.GetTimestamp() - tPlain0;

                // Reused, not freshly allocated: a vocabulary row is 500-600 KB, so
                // one array per decoded token is a large-object allocation per token
                // on a path that is otherwise a few milliseconds. The non-speculative
                // route hands back the model's own buffer under the same
                // consume-before-the-next-forward contract; this buffer is written
                // only here, so it survives until the next plain step.
                _plainOut ??= new float[_vocab];
                Array.Copy(_stepLogits, _plainOut, _vocab);
                float[] plainLogits = _plainOut;
                // A step whose DRAFT ran and proposed nothing (the head under its gate)
                // is speculation's cost - the head pass is in it - not the plain
                // baseline's: charged to plain it inflated the baseline every verdict
                // is measured against, and a head that stayed under its gate never
                // filled the probe's speculative quota, so the round never closed and
                // the head ran on every step unparked.
                // That holds for a matchless n-gram lookup too, although it paid no head
                // pass: its step still ran on the SPECULATIVE path's plain step, which on
                // a trunk whose families differ (Qwen 3.5) is the dearer one, and that is
                // exactly the cost a parked step would not pay. Counting such steps as
                // plain instead inflated the baseline with them and let n-gram run
                // through prose unparked: Qwen prose fell from 47 to 38 tok/s.
                RecordStep(speculated: kMax > 0, tokensEmitted: 1, tStep0);
                return new SpeculativeStepOutcome
                {
                    AcceptedCount = 0,
                    NextToken = -1,
                    NextLogits = plainLogits,
                    UsedSpeculation = false,
                };
            }

            // VERIFY: one batched trunk forward over [lastToken, d1..dK].
            Stats.VerifySteps++;
            _plainRun = 0;
            int k = _draftTokens.Count;
            int[] batch = new int[k + 1];
            batch[0] = lastToken;
            for (int i = 0; i < k; i++)
                batch[i + 1] = _draftTokens[i];

            long tSnap0 = Stopwatch.GetTimestamp();
            _trunk.SnapshotRecurrentState();
            long tVerify0 = Stopwatch.GetTimestamp();
            Stats.SnapshotTicks += tVerify0 - tSnap0;
            // Supplying drawGreedy is an explicit promise that each row reduces to
            // raw argmax after the caller's forced-token/stop checks. Legacy callers
            // always receive full logits, including when a target declines the hook.
            bool greedyVerify = drawGreedy != null && _trunk.TryForwardGreedyVerify(batch, _verifyH, _verifyDrawn);
            if (!greedyVerify)
                _trunk.Forward(batch, _verifyH, _verifyLogits, allLogitsRows: true);
            Stats.VerifyTicks += Stopwatch.GetTimestamp() - tVerify0;

            int m = 0;
            int nextToken;
            while (true)
            {
                int drawn;
                if (greedyVerify)
                    drawn = drawGreedy!(_verifyDrawn[m]);
                else
                {
                    Array.Copy(_verifyLogits, (long)m * _vocab, _rowLogits, 0, _vocab);
                    drawn = drawNext(_rowLogits);
                }
                if (m < k && drawn == _draftTokens[m])
                {
                    onDraftAccepted?.Invoke(drawn);
                    m++;
                    continue;
                }
                nextToken = drawn;
                break;
            }

            Stats.TokensDrafted += k;
            Stats.TokensAccepted += m;
            Telemetry.TensorSharpMetrics.SpeculativeTokensDrafted.Add(k, _architectureTag);
            Telemetry.TensorSharpMetrics.SpeculativeTokensAccepted.Add(m, _architectureTag);
            RecordAcceptedLength(k, m);

            // Tell the trunk the accept count BEFORE deciding how to roll back: a
            // trunk that deferred post-verify state until this was known settles it
            // now, and that is what can turn the partial-acceptance branch below from
            // a whole second forward into a position rewind.
            _trunk.OnVerifyAccepted(m, k);

            // The tokens this step commits to the trunk: the verify batch's
            // accepted prefix plus the token it started from.
            int[] keep = new int[m + 1];
            Array.Copy(batch, keep, m + 1);

            if (m < k)
            {
                // Partial acceptance. Fast path: if the verify already persisted
                // reusable KV for the accepted prefix (no recurrent state), just keep
                // those writes and advance the position - the kept-prefix re-forward
                // is redundant (it would recompute byte-identical KV). This is the
                // dominant rollback cost on long contexts. Otherwise roll back to the
                // pre-verify checkpoint and re-advance over the kept prefix.
                Stats.RollbackSteps++;
                long tRoll0 = Stopwatch.GetTimestamp();
                if (!_trunk.TryCommitVerifiedPrefix(position + m + 1))
                {
                    _trunk.Rollback(position);
                    _trunk.Forward(keep, null, _stepLogits, allLogitsRows: false);
                }
                Stats.RollbackTicks += Stopwatch.GetTimestamp() - tRoll0;
            }

            // Hand the kept tokens to the speculator with their exact trunk
            // hidden states.
            {
                long tCatch0 = Stopwatch.GetTimestamp();
                if (NeedsHidden)
                {
                    Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                    if (m > 0)
                        Array.Copy(_verifyH, 0, _catchUpH, _hidden, (long)m * _hidden);
                }
                _speculator.Commit(keep, _catchUpH, position);
                Stats.CatchUpTicks += Stopwatch.GetTimestamp() - tCatch0;
            }

            if (NeedsHidden)
                Array.Copy(_verifyH, (long)m * _hidden, _pendingH, 0, _hidden);

            float[]? nextLogits = null;
            if (!greedyVerify)
            {
                nextLogits = new float[_vocab];
                Array.Copy(_verifyLogits, (long)m * _vocab, nextLogits, 0, _vocab);
            }
            // m accepted drafts plus the corrected/bonus token.
            RecordStep(speculated: true, tokensEmitted: m + 1, tStep0);
            return new SpeculativeStepOutcome
            {
                AcceptedCount = m,
                NextToken = nextToken,
                NextLogits = nextLogits,
                UsedSpeculation = true,
            };
        }

        /// <summary>
        /// Batched step, first half: draft for this sequence ahead of a verify that
        /// serves several sequences at once (<see cref="IBatchedSpeculativeTarget"/>).
        /// The caller binds this sequence's drafter state first. Returns the number of
        /// drafts in <paramref name="draftsOut"/>; 0 (no carry yet, no room) means the
        /// sequence rides the verify with its anchor alone. The cost governor is not
        /// consulted: beside other sequences a window costs a few columns of a shared
        /// sweep, and the plain alternative is a step of the same sweep.
        /// </summary>
        public int ProposeForBatch(int lastToken, int position, int kMax, List<int> draftsOut,
            Action<float[], IReadOnlyList<int>>? adjustDraftLogits = null)
        {
            ArgumentNullException.ThrowIfNull(draftsOut);
            draftsOut.Clear();
            kMax = Math.Min(kMax, MaxDraftTokens);
            kMax = Math.Min(kMax, _model.MaxContextLength - position - 2);
            if (!CarryValid || kMax <= 0)
                return 0;
            long t0 = Stopwatch.GetTimestamp();
            _speculator.Propose(
                new DraftContext
                {
                    LastToken = lastToken,
                    Position = position,
                    MaxTokens = kMax,
                    CarryHidden = _pendingH,
                    AdjustLogits = adjustDraftLogits,
                },
                draftsOut);
            if (draftsOut.Count > kMax)
                throw new InvalidOperationException(
                    $"The {_speculator.Name} speculator proposed {draftsOut.Count} tokens "
                    + $"for a verification window limited to {kMax} drafts.");
            Stats.DraftTicks += Stopwatch.GetTimestamp() - t0;
            return draftsOut.Count;
        }

        /// <summary>True when this execution's drafts can come from a block drafted for
        /// several sequences in one pass (<see cref="IBatchedSpeculativeTarget.TryDraftBatch"/>).</summary>
        public bool SupportsBatchDraft => NeedsHidden && _speculator is BlockDraftSpeculator;

        /// <summary>
        /// Batched step, first half when drafting for several sequences in one pass:
        /// whether this sequence drafts at all this step and, if so, its carried
        /// feature row and draft cap. The same gates as <see cref="ProposeForBatch"/>.
        /// </summary>
        public bool TryPrepareBatchDraft(int position, int kMax, [NotNullWhen(true)] out float[]? carry, out int maxTokens)
        {
            carry = null;
            maxTokens = Math.Min(Math.Min(kMax, MaxDraftTokens), _model.MaxContextLength - position - 2);
            if (!SupportsBatchDraft || !CarryValid || maxTokens <= 0)
                return false;
            carry = _pendingH;
            return true;
        }

        /// <summary>Apply the drafter's gate to a block drafted for this sequence by a
        /// multi-sequence pass. Returns the number of drafts in
        /// <paramref name="draftsOut"/>.</summary>
        public int AcceptBatchDraft(int[] blockTokens, float[] blockConf, int maxTokens, List<int> draftsOut, long draftTicks)
        {
            ArgumentNullException.ThrowIfNull(draftsOut);
            draftsOut.Clear();
            Stats.DraftTicks += draftTicks;
            if (_speculator is not BlockDraftSpeculator block)
                return 0;
            return block.SelectDrafts(blockTokens, blockConf, Math.Min(blockTokens.Length, blockConf.Length), maxTokens, draftsOut);
        }

        /// <summary>One verify window's accepted drafts, by how many it drafted
        /// (tensorsharp.speculative.accepted_length): the distribution behind the drafted and accepted totals.</summary>
        private void RecordAcceptedLength(int drafted, int accepted)
        {
            var histogram = Telemetry.TensorSharpMetrics.SpeculativeAcceptedLength;
            if (histogram.Enabled && drafted > 0)
                histogram.Record(accepted, _architectureTag,
                    new KeyValuePair<string, object?>(Telemetry.TelemetryAttributes.SpeculativeDrafted, drafted));
        }

        /// <summary>
        /// Batched step, second half: the trunk verified [<paramref name="lastToken"/>,
        /// drafts] at <paramref name="position"/> and kept the anchor plus
        /// <paramref name="accepted"/> drafts. Hands the kept tokens to the speculator
        /// with their trunk taps (<paramref name="rowFeatures"/> from
        /// <paramref name="rowOffset"/>, one SpecFeatureSize row per verify row) and
        /// carries the last kept row into the next draft. The caller binds this
        /// sequence's drafter state first and has already advanced the trunk.
        /// </summary>
        public void CommitBatch(int lastToken, int position, IReadOnlyList<int> drafts, int accepted,
            float[]? rowFeatures, long rowOffset, long verifyTicks)
        {
            ArgumentNullException.ThrowIfNull(drafts);
            if (accepted < 0 || accepted > drafts.Count)
                throw new ArgumentOutOfRangeException(nameof(accepted));
            int k = drafts.Count;
            if (k > 0)
            {
                Stats.VerifySteps++;
                Stats.TokensDrafted += k;
                Stats.TokensAccepted += accepted;
                Telemetry.TensorSharpMetrics.SpeculativeTokensDrafted.Add(k, _architectureTag);
                Telemetry.TensorSharpMetrics.SpeculativeTokensAccepted.Add(accepted, _architectureTag);
                RecordAcceptedLength(k, accepted);
                if (accepted < k)
                    Stats.RollbackSteps++;
                Stats.VerifyTicks += verifyTicks;
                _plainRun = 0;
            }
            else
            {
                Stats.PlainSteps++;
                Stats.PlainTicks += verifyTicks;
            }

            int[] keep = new int[accepted + 1];
            keep[0] = lastToken;
            for (int i = 0; i < accepted; i++)
                keep[i + 1] = drafts[i];
            long tCatch0 = Stopwatch.GetTimestamp();
            if (!NeedsHidden || rowFeatures == null)
            {
                _speculator.Commit(keep, null, position);
                _carryValid = !NeedsHidden;
            }
            else
            {
                // Row k pairs token k with the taps of the token before it: the carried
                // row for the anchor, then the verify rows of the kept prefix.
                bool carryKnown = CarryValid;
                Array.Copy(_pendingH, 0, _catchUpH, 0, _hidden);
                if (accepted > 0)
                    Array.Copy(rowFeatures, rowOffset, _catchUpH, _hidden, (long)accepted * _hidden);
                _speculator.Commit(keep, carryKnown ? _catchUpH : null, position);
                Array.Copy(rowFeatures, rowOffset + (long)accepted * _hidden, _pendingH, 0, _hidden);
                _carryValid = true;
            }
            Stats.CatchUpTicks += Stopwatch.GetTimestamp() - tCatch0;
        }

        public void Dispose() => _speculator.Dispose();

        private readonly int _governorWinsAtStart, _governorLossesAtStart, _governorParkedAtStart;

        private void RecordStep(bool speculated, int tokensEmitted, long tStep0)
        {
            Governor.Record(speculated, tokensEmitted, Stopwatch.GetTimestamp() - tStep0);
            Stats.PlainMsPerToken = Governor.PlainMsPerToken;
            Stats.SpecMsPerToken = Governor.SpecMsPerToken;
            Stats.GovernorWins = Governor.Wins - _governorWinsAtStart;
            Stats.GovernorLosses = Governor.Losses - _governorLossesAtStart;
            Stats.GovernorParkedSteps = Governor.ParkedSteps - _governorParkedAtStart;
        }

        [MemberNotNull(nameof(_chunkH), nameof(_lastRowH))]
        private void EnsureChunkBuffers(int chunkLen)
        {
            long need = (long)chunkLen * _hidden;
            if (_chunkH == null || _chunkH.Length < need)
                _chunkH = new float[need];
            _lastRowH ??= new float[_hidden];
        }
    }
}
