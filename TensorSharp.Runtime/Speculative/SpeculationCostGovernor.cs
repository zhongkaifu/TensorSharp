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
using System.Diagnostics;

namespace TensorSharp.Runtime.Speculative
{
    /// <summary>
    /// Decides, at run time and from measurement alone, whether speculation is
    /// currently worth doing - and parks it when it is not.
    ///
    /// Speculation is only ever a SPEED optimization (the emitted stream is
    /// identical either way), so it must never make decoding slower. Whether it
    /// pays off is a property of the (model, drafter, backend) TRIPLE, not of
    /// the code: on Muse-Glimmer-30B (a sparse MoE where a decode step only
    /// touches the active experts) the 1.5 GB dense DFlash drafter costs roughly
    /// half a trunk step per drafted token, and measured end to end at every
    /// window size from 1 to 15 it ran 18-31% SLOWER than plain greedy despite a
    /// 70% acceptance rate. Acceptance rate alone cannot see that: it says
    /// nothing about what a draft COSTS relative to the trunk.
    ///
    /// So measure both and let the numbers decide. A round measures speculation
    /// first (so the very first decode step still speculates), then takes a
    /// short plain baseline, then judges. A win holds for
    /// <see cref="WinRecheckInterval"/> steps; a loss parks drafting for a
    /// backed-off interval and re-probes, so a pair that becomes profitable
    /// later (longer context, different prompt) is picked back up. Overhead when
    /// speculation is a loss is bounded by one probe per park interval.
    ///
    /// Kept apart from <see cref="SpeculativeExecution"/> on purpose: this is a
    /// measurement POLICY, and it is the piece most likely to be tuned,
    /// replaced or switched off per deployment without touching the
    /// draft/verify protocol.
    /// </summary>
    public sealed class SpeculationCostGovernor
    {
        /// <summary>Speculative steps measured per round before judging speculation,
        /// on the first round and after a winning verdict.</summary>
        private const int ProbeSpecSteps = 8;
        /// <summary>
        /// Speculative steps a RE-PROBE takes after a losing verdict. A probe is paid
        /// in speculative steps, and on a pairing that loses those are the dear ones:
        /// Gemma 4 E4B-IQ4_XS with its draft head spends about four plain steps per
        /// speculative one, so an eight-step re-probe every park interval kept prose
        /// 30-50% slower than plain decoding while nominally parked. Three steps still
        /// see a repetitive stretch (an accepted window shows in one), and a win
        /// verdict restores the full probe the next time speculation is doubted.
        /// </summary>
        private const int ReprobeSpecSteps = 3;
        private int ProbeStepsThisRound => _consecutiveLosses > 0 ? ReprobeSpecSteps : ProbeSpecSteps;

        /// <summary>Plain steps measured per round, for the baseline to compare against.</summary>
        private const int CalibrationPlainSteps = 3;

        /// <summary>
        /// A probe ends early, as a loss, when its first this-many measured
        /// speculative steps have all accepted nothing: every one of them cost a verify (2-4x a
        /// plain step) for a single token, and on free prose an n-gram drafter
        /// accepts nothing for whole turns (30 drafted, 0 accepted over 10 verifies
        /// on Qwen 3.5-9B). Half the probe, same verdict.
        /// </summary>
        private const int ZeroAcceptProbeSteps = 4;

        /// <summary>Longest a losing verdict may park drafting for. With a shared
        /// governor the park carries across a chat's turns, so a pairing that keeps
        /// losing settles at one three-step re-probe per 256 tokens (~3% on a trunk
        /// whose speculative step costs four plain ones) instead of one per 64.</summary>
        private const int ParkedProbeInterval = 256;

        /// <summary>
        /// Park length after the FIRST losing verdict. A verdict formed right after
        /// a prefill is the least trustworthy one the governor ever makes, so the
        /// penalty for getting it wrong is capped low and only backs off toward
        /// <see cref="ParkedProbeInterval"/> if speculation keeps losing.
        /// </summary>
        // 32 rather than 16: on a losing turn every re-probe is three dear
        // speculative steps, and 16/32/64/128 fitted four of them into a 160-token
        // prose turn (55 tok/s against 65 plain on E4B-IQ4_XS); 32/64/128 fits two.
        private const int FirstParkInterval = 32;

        /// <summary>
        /// Steps a winning verdict holds for before the next measuring round. Short,
        /// because a round taken while speculation WINS costs almost nothing - its
        /// speculative samples are productive steps and only the three plain
        /// baseline steps are overhead - while a wrong win (a probe that happened to
        /// land on a repetitive stretch, the thinking preamble say, ahead of free
        /// prose) used to hold for 512 steps: E4B-IQ4_XS with its draft head ran a
        /// prose turn at 24 tok/s against 65 plain under one.
        /// </summary>
        private const int WinRecheckInterval = 64;

        /// <summary>
        /// Speculative samples watched WHILE a winning verdict holds. A hold used to
        /// be blind: a probe that landed on the predictable thinking preamble
        /// ("The user is asking me to write three short paragraphs...") won
        /// honestly, and the hold then rode through 64 steps of free prose at 17%
        /// acceptance - E4B-IQ4_XS prose at 30 tok/s against 65 plain, twice in
        /// three runs. The hold now keeps timing its speculative steps against the
        /// plain baseline the round measured, and parks the moment the last
        /// <see cref="WinWindow"/> of them average a loss; a wrong win costs about
        /// eight steps instead of sixty-four.
        /// </summary>
        private const int WinWindow = 8;
        private readonly long[] _winTicks = new long[WinWindow];
        private readonly int[] _winTokens = new int[WinWindow];
        private int _winCount, _winHead;

        /// <summary>
        /// Speculation must not be more than this much slower to stay on. This is a
        /// tolerance on a NOISY estimator, not a precision knob: the sample is 8
        /// steps taken at the least predictable point of a response, so a few
        /// percent either way carries no information. It was 1.02, which let
        /// ordinary sampling noise park a drafter that was measurably 2.7x faster
        /// on the very next probe.
        /// </summary>
        private const double SpecWinMargin = 1.15;

        // Ratio of SUMS, not a mean of per-step rates. A speculative step emits
        // between 1 and MaxDraftTokens+1 tokens; normalising each step to ms/token
        // BEFORE averaging weights a fully-rejected step (1 token, full step cost)
        // exactly as heavily as a fully-accepted one (16 tokens, barely more cost),
        // which over-states speculation's cost and only speculation's - plain steps
        // always emit one token, so their mean of rates already equals their
        // aggregate. On a realistic 8-step probe (4 steps x 16 tokens, 4 steps x 1)
        // the old estimator read 29.4 ms/token where the true figure was 6.8.
        private long _plainTicks, _specTicks;
        private long _plainTokens, _specTokens;
        private int _plainSamples, _specSamples;

        // The single most expensive speculative sample of the round (by ms/token),
        // kept as its raw (ticks, tokens) so the verdict can drop the WHOLE step.
        // The verify graph is rebuilt whenever the batch shape or the padded KV
        // window changes, and both happen DURING a probe (k varies per step, and 8
        // speculative steps can advance the position across a window boundary), so
        // one sample in a round routinely carries a graph build that never recurs.
        private long _specWorstTicks;
        private int _specWorstTokens;

        // Whether the first sample of each side has already been discarded this
        // round. The first speculative step after a prefill pays four one-off graph
        // builds (trunk verify shape, drafter inject at n_rows=1, the draft block,
        // and possibly a KV-capacity grow that drops every persistent graph); the
        // first plain step after a run of verify batches finds the 1-row trunk
        // graph cold for the same reason.
        private bool _specFirstSkipped, _plainFirstSkipped;

        private int _parkedRemaining;
        private int _recheckRemaining;
        private int _consecutiveLosses;
        private bool _specWins = true;

        /// <summary>
        /// Long-run evidence: every measured step's (ticks, tokens) on each side, token-weighted and decayed by this
        /// much per further step of the same side (a half-life of ~23 steps). Verdicts compare these, not the round
        /// alone: a speculative step emits between one token and a whole window, so eight steps - three on a re-probe -
        /// are a noisy sample, and on Qwen3.8 Flash Next (M5 Pro, DFlash2) single unlucky windows parked speculation
        /// for up to 371 steps of a request whose speculative steps cost 13-27 ms a token against 30-35 plain. Greedy
        /// long generation 40-43 -> 46-47 tok/s without that, sampled (temperature 1.0) decode after a 4K prompt
        /// 34.5 -> 45. A loss that holds still flips the evidence within a dozen steps: a drafter at a quarter of the
        /// acceptance and four times a plain step's cost parks after ~7, one barely over the margin after ~11.
        /// </summary>
        private const double EvidenceDecay = 0.97;
        private double _evSpecTicks, _evSpecTokens, _evPlainTicks, _evPlainTokens;

        private void AddSpecEvidence(long ticks, int tokens)
        {
            _evSpecTicks = _evSpecTicks * EvidenceDecay + ticks;
            _evSpecTokens = _evSpecTokens * EvidenceDecay + tokens;
        }

        private void AddPlainEvidence(long ticks, int tokens)
        {
            _evPlainTicks = _evPlainTicks * EvidenceDecay + ticks;
            _evPlainTokens = _evPlainTokens * EvidenceDecay + tokens;
        }

        private double EvidenceSpecMs => _evSpecTokens > 0 ? _evSpecTicks * 1000.0 / Stopwatch.Frequency / _evSpecTokens : 0;
        private double EvidencePlainMs => _evPlainTokens > 0 ? _evPlainTicks * 1000.0 / Stopwatch.Frequency / _evPlainTokens : 0;

        /// <summary>
        /// Measure speculation against plain decoding at runtime and skip drafting
        /// while it is measurably slower. On by default; set false to force the
        /// drafter on for A/B measurement (TS_SPEC_ADAPTIVE=0 does the same from
        /// the environment, which is how the plain-baseline cost itself is
        /// measured - a round's baseline steps are plain decodes, and they are not
        /// free).
        /// </summary>
        public bool Enabled { get; set; } = DefaultEnabled;
        public string SelectionSource { get; internal set; } = "legacy-default";

        internal static readonly string? AdaptiveEnvironmentSnapshot = Environment.GetEnvironmentVariable("TS_SPEC_ADAPTIVE");
        private static readonly bool DefaultEnabled = AdaptiveEnvironmentSnapshot != "0";

        /// <summary>True while a losing verdict is actively suppressing drafting
        /// (as opposed to a round merely taking its plain baseline). Only these
        /// steps are reported as "parked".</summary>
        public bool IsParked => _parkedRemaining > 0;

        /// <summary>
        /// A new request is starting on this (shared) governor. The park carried
        /// over from the previous turn was earned on THAT turn's text; the next
        /// turn's may be nothing like it - a quote after prose reached 99%
        /// acceptance and still ran 122 of its 135 steps plain because a 128-step
        /// park was still counting down. Cap what is left at the first park
        /// interval so the new turn re-probes (three speculative steps) within its
        /// first thirty-two; the losses already counted keep the backoff, so a
        /// pairing that keeps losing still parks longer each time.
        /// </summary>
        public void NewSequence()
        {
            if (_parkedRemaining > FirstParkInterval)
                _parkedRemaining = FirstParkInterval;
            // The backoff was earned on the previous turn's text too. Kept at one
            // loss - so the re-probe stays the short one - it restarts at the first
            // interval: with it left at five, one noisy three-step re-probe on a quote
            // turn parked the remaining 200 steps of a turn that would have run 85
            // tok/s against 65.
            if (_consecutiveLosses > 1)
                _consecutiveLosses = 1;
            // Samples of the previous request's text say nothing about this one.
            ResetRound();
        }

        /// <summary>
        /// Plain steps the request took OUTSIDE its speculative execution - a
        /// concurrent interlude serves it on the batched or serial fused path - still
        /// elapse a park or a hold, so a resumed context does not start a fresh round.
        /// </summary>
        public void NoteExternalPlainSteps(int steps)
        {
            if (!Enabled || steps <= 0)
                return;
            int parked = Math.Min(steps, _parkedRemaining);
            _parkedRemaining -= parked;
            ParkedSteps += parked;
            _recheckRemaining = Math.Max(0, _recheckRemaining - (steps - parked));
        }

        /// <summary>Verdicts so far: rounds (and held wins) that ended in a win / a loss,
        /// and steps spent parked - the per-turn stats line reports them.</summary>
        public int Wins { get; private set; }
        public int Losses { get; private set; }
        public int ParkedSteps { get; private set; }

        /// <summary>Measured ms per emitted token for the current round, plain
        /// side. 0 until the round has a sample.</summary>
        public double PlainMsPerToken { get; private set; }

        /// <summary>Measured ms per emitted token for the current round,
        /// speculative side. 0 until the round has a sample.</summary>
        public double SpecMsPerToken { get; private set; }

        /// <summary>Whether this step may draft.</summary>
        public bool AllowsSpeculation()
        {
            if (!Enabled)
                return true;
            if (_parkedRemaining > 0)
                return false;
            if (_recheckRemaining > 0)
                return _specWins;
            if (_specSamples < ProbeStepsThisRound)
                return true;                        // measuring speculation
            if (_plainSamples < CalibrationPlainSteps)
                return false;                       // measuring the plain baseline
            return _specWins;
        }

        /// <summary>Record one completed decode step.</summary>
        public void Record(bool speculated, int tokensEmitted, long elapsedTicks)
        {
            if (!Enabled || tokensEmitted <= 0)
                return;

            if (_parkedRemaining > 0)
            {
                _parkedRemaining--;
                ParkedSteps++;
                // A parked step IS the plain baseline: it runs the same decode the
                // non-speculative engine would have run for this token. Sampling it
                // costs nothing and is the only measurement of plain cost taken
                // outside a graph-family transition, so the next round starts with a
                // baseline drawn from 32-256 clean tokens instead of three dear
                // calibration steps. The first one after speculation is skipped:
                // that is the step that pays the transition.
                if (!speculated)
                    RecordPlainSample(elapsedTicks, tokensEmitted);
                return;
            }
            if (_recheckRemaining > 0)
            {
                _recheckRemaining--;
                if (speculated && PlainMsPerToken > 0)
                {
                    _winTicks[_winHead] = elapsedTicks;
                    _winTokens[_winHead] = tokensEmitted;
                    _winHead = (_winHead + 1) % WinWindow;
                    if (_winCount < WinWindow)
                        _winCount++;
                    AddSpecEvidence(elapsedTicks, tokensEmitted);
                    if (_winCount == WinWindow)
                    {
                        // Judged on the long-run evidence the window has just been added to, not on the window alone.
                        SpecMsPerToken = EvidenceSpecMs;
                        if (SpecMsPerToken > EvidencePlainMs * SpecWinMargin)
                        {
                            // The win stopped being one: park with the usual backoff.
                            _specWins = false;
                            Losses++;
                            _consecutiveLosses++;
                            _parkedRemaining = Math.Min(ParkedProbeInterval, FirstParkInterval << Math.Min(_consecutiveLosses - 1, 5));
                            _recheckRemaining = 0;
                            ResetRound();
                        }
                    }
                }
                return;
            }

            if (speculated)
            {
                if (!_specFirstSkipped) { _specFirstSkipped = true; return; }
                _specTicks += elapsedTicks;
                _specTokens += tokensEmitted;
                _specSamples++;
                AddSpecEvidence(elapsedTicks, tokensEmitted);
                if (_specWorstTokens == 0 ||
                    MsPerToken(elapsedTicks, tokensEmitted) > MsPerToken(_specWorstTicks, _specWorstTokens))
                {
                    _specWorstTicks = elapsedTicks;
                    _specWorstTokens = tokensEmitted;
                }
                SpecMsPerToken = MsPerToken(_specTicks, _specTokens);
                // Nothing accepted in the first ZeroAcceptProbeSteps measured steps: a
                // loss, and the rest of the probe would only confirm it.
                if (_specSamples == ZeroAcceptProbeSteps && _specTokens == _specSamples
                    && _specSamples < ProbeStepsThisRound)
                {
                    _specWins = false;
                    Losses++;
                    _consecutiveLosses++;
                    _parkedRemaining = Math.Min(ParkedProbeInterval, FirstParkInterval << Math.Min(_consecutiveLosses - 1, 5));
                    _recheckRemaining = 0;
                    ResetRound();
                    return;
                }
            }
            else
            {
                RecordPlainSample(elapsedTicks, tokensEmitted);
            }

            if (_specSamples < ProbeStepsThisRound || _plainSamples < CalibrationPlainSteps)
                return;

            // Trim the worst speculative sample (a graph rebuild that will not
            // recur) before comparing. Only when it leaves something to compare.
            double specMs = MsPerToken(_specTicks, _specTokens);
            if (_specSamples > 1)
            {
                long keptTicks = _specTicks - _specWorstTicks;
                long keptTokens = _specTokens - _specWorstTokens;
                if (keptTicks > 0 && keptTokens > 0)
                    specMs = Math.Min(specMs, MsPerToken(keptTicks, keptTokens));
            }

            // A win on either the round (worst sample trimmed) or the long-run evidence: a short re-probe that lands on
            // an unlucky stretch does not overrule the steps before it, and a real loss shows in both.
            double plainMs = MsPerToken(_plainTicks, _plainTokens);
            _specWins = specMs <= plainMs * SpecWinMargin
                || (EvidenceSpecMs > 0 && EvidencePlainMs > 0 && EvidenceSpecMs <= EvidencePlainMs * SpecWinMargin);
            if (_specWins) Wins++; else Losses++;
            _consecutiveLosses = _specWins ? 0 : _consecutiveLosses + 1;
            _parkedRemaining = _specWins
                ? 0
                : Math.Min(ParkedProbeInterval, FirstParkInterval << Math.Min(_consecutiveLosses - 1, 5));
            _recheckRemaining = _specWins ? WinRecheckInterval : 0;
            ResetRound();
        }

        /// <summary>
        /// One plain-side sample. The first of a round is dropped: a plain step that
        /// follows a speculative one pays whatever the backend charges to change
        /// graph family, and charging a transition to the steady-state cost of plain
        /// decoding is what let a 5x regression read as a win.
        /// </summary>
        private void RecordPlainSample(long elapsedTicks, int tokensEmitted)
        {
            if (!_plainFirstSkipped) { _plainFirstSkipped = true; return; }
            _plainTicks += elapsedTicks;
            _plainTokens += tokensEmitted;
            _plainSamples++;
            AddPlainEvidence(elapsedTicks, tokensEmitted);
            PlainMsPerToken = MsPerToken(_plainTicks, _plainTokens);
        }

        /// <summary>
        /// Drop every verdict and start measuring from scratch. A park decided on
        /// one turn describes that turn's context length and prompt, and carrying
        /// it into the next request means a fresh generation can start with
        /// drafting disabled for no reason.
        /// </summary>
        public void Reset()
        {
            ResetRound();
            _parkedRemaining = _recheckRemaining = 0;
            _consecutiveLosses = 0;
            _specWins = true;
            Wins = Losses = ParkedSteps = 0;
            _evSpecTicks = _evSpecTokens = _evPlainTicks = _evPlainTokens = 0;
            // A fresh governor has measured nothing; leaving the last run's
            // rates behind would make a per-turn log report numbers from the
            // previous turn's context length.
            PlainMsPerToken = SpecMsPerToken = 0;
        }

        private void ResetRound()
        {
            _specSamples = _plainSamples = 0;
            _specTicks = _plainTicks = 0;
            _specTokens = _plainTokens = 0;
            _specWorstTicks = 0;
            _specWorstTokens = 0;
            _specFirstSkipped = _plainFirstSkipped = false;
            _winCount = _winHead = 0;
        }

        private static double MsPerToken(long ticks, long tokens) =>
            tokens > 0 ? ticks * 1000.0 / Stopwatch.Frequency / tokens : 0.0;
    }
}
