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
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    /// <summary>
    /// The standalone speculative generate loop: KV reset, chunked prompt
    /// prefill, and the emit/stop/penalty-history bookkeeping around
    /// <see cref="SpeculativeExecution"/> — the shared draft/verify/rollback core
    /// that the engine's <c>BatchExecutor</c> also drives, one step per scheduler
    /// iteration. Used by the CLI, the tests and offline callers.
    ///
    /// Algorithm-agnostic: whatever <see cref="ISpeculator"/> it is given does
    /// the drafting — a per-token NextN/MTP head (Qwen 3.5/3.6, GLM-5.2, Gemma 4),
    /// a block drafter (DeepSeek V4 DSpark, Muse-Glimmer DFlash), or the
    /// weight-free n-gram speculator, which needs no draft head at all.
    ///
    /// Verification draws either with argmax (<see cref="GenerateGreedy"/>) or
    /// with the caller's sampler (<see cref="GenerateSampled"/>). Either way every
    /// emitted token comes from a trunk row, so the output stream is the one plain
    /// decoding would have produced — greedily up to batched-vs-sequential
    /// floating point drift, and in distribution under a sampler.
    /// Single-sequence; the caller owns the model's KV cache lifecycle.
    /// </summary>
    public sealed class SpeculativeDecoder
    {
        private readonly ISpeculativeTarget _model;
        private readonly SpeculativeExecution _exec;
        private readonly int _vocab;
        private readonly List<int> _acceptedScratch = new();
        // Penalty history for a sampled run (see GenerateSampledFrom). Reused
        // across generations; cleared at the start of each.
        private readonly List<int> _historyScratch = new();

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp n_max).</summary>
        public int MaxDraftTokens => _exec.MaxDraftTokens;

        /// <summary>
        /// The drafter's confidence gate; drafting stops at the first token below
        /// it. What the number MEANS is the algorithm's business — see
        /// <see cref="ISpeculator.MinDraftProb"/>.
        /// </summary>
        public float MinDraftProb
        {
            get => _exec.MinDraftProb;
            set => _exec.MinDraftProb = value;
        }

        /// <summary>Prompt prefill chunk size (bounds the h-capture buffers).</summary>
        public int PrefillChunkSize { get; set; } = 512;

        // Statistics (reset by Reset()).
        public long TokensDrafted => _exec.Stats.TokensDrafted;
        public long TokensAccepted => _exec.Stats.TokensAccepted;
        public long VerifySteps => _exec.Stats.VerifySteps;
        public long PlainSteps => _exec.Stats.PlainSteps;
        public long RollbackSteps => _exec.Stats.RollbackSteps;
        /// <summary>Steps the cost governor forced to plain decoding.</summary>
        public long ParkedSteps => _exec.Stats.ParkedSteps;
        public double PlainMsPerToken => _exec.Stats.PlainMsPerToken;
        public double SpecMsPerToken => _exec.Stats.SpecMsPerToken;
        /// <summary>Set false to force drafting on regardless of measured cost.</summary>
        public bool AdaptiveSpeculation
        {
            get => _exec.AdaptiveSpeculation;
            set => _exec.AdaptiveSpeculation = value;
        }
        public double AcceptanceRate => _exec.Stats.AcceptanceRate;

        /// <summary>Wall-clock phase breakdown of the speculative decode loop
        /// (draft / verify / snapshot / rollback / catch-up / plain).</summary>
        public SpeculationStats Stats => _exec.Stats;

        /// <summary>
        /// The trunk position this decoder can be resumed at, or -1 when it cannot.
        /// A per-token draft head chains from the hidden state of the token before
        /// the one it drafts, so continuing a conversation by prefilling only the
        /// new suffix is sound exactly when that carry-in still describes the token
        /// the trunk now ends on. It does after a prefill and after a generation
        /// that ran to its end; it does NOT after a run that stopped part-way
        /// through a verified window, because the trunk is then rewound below the
        /// row the carry-in came from.
        ///
        /// Resuming anyway is silent: the emitted stream stays correct (verification
        /// is trunk-driven), but the draft head's KV picks up one row built from the
        /// wrong hidden state and acceptance decays from there. A caller that
        /// extends a cache across turns must check this before prefilling a suffix.
        /// </summary>
        public int CarryPosition { get; private set; } = -1;

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent in prompt prefill.</summary>
        public double LastPrefillSeconds { get; private set; }

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent past prefill.</summary>
        public double LastDecodeSeconds { get; private set; }

        /// <summary>
        /// Decode <paramref name="model"/> with an explicit algorithm - the
        /// general form. Any <see cref="ISpeculator"/> works, including one that
        /// needs no draft head at all (<see cref="NGramSpeculator"/>).
        /// </summary>
        public SpeculativeDecoder(ISpeculativeTarget model, ISpeculator speculator)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            ArgumentNullException.ThrowIfNull(speculator);
            _exec = new SpeculativeExecution(model, speculator);
            _vocab = model.Config.VocabSize;
        }

        /// <summary>
        /// Decode <paramref name="model"/> with the algorithm
        /// <paramref name="options"/> selects (default: whatever drafter the
        /// checkpoint carries).
        /// </summary>
        /// <exception cref="ArgumentException">No registered algorithm can serve
        /// this model; the message says why.</exception>
        public SpeculativeDecoder(ISpeculativeTarget model, SpeculationOptions options)
            : this(model, Resolve(model, options))
        {
        }

        /// <summary>
        /// Convenience for the common case: use the checkpoint's own draft head
        /// with a window of <paramref name="maxDraftTokens"/>.
        ///
        /// The default window is 8. A confidence gate stops drafting at the
        /// first low-confidence token, so a longer window only extends confident
        /// streaks — measured 1.21x vs 1.08x (window 4) on Qwen3.6-35B-A3B
        /// ggml_cpu at unchanged 86% acceptance; neutral on 27B ggml_cuda.
        /// </summary>
        public SpeculativeDecoder(ISpeculativeModel model,
            int maxDraftTokens = SpeculationOptions.DefaultMaxDraftTokens)
            : this(model, new SpeculationOptions { Enabled = true, MaxDraftTokens = maxDraftTokens })
        {
        }

        private static ISpeculator Resolve(ISpeculativeTarget model, SpeculationOptions options)
        {
            ArgumentNullException.ThrowIfNull(model);
            var speculator = SpeculatorRegistry.Create(model, options, out string declineReason);
            if (speculator == null)
                throw new ArgumentException($"Speculative decoding is unavailable: {declineReason}", nameof(model));
            return speculator;
        }

        /// <summary>The algorithm this decoder drafts with.</summary>
        public ISpeculator Speculator => _exec.Speculator;

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset()
        {
            _exec.Reset();
            CarryPosition = -1;
        }

        /// <summary>
        /// Start the statistics and the cost governor over while keeping the
        /// carry-in hidden state, for a caller that reports per turn but extends
        /// the same KV cache across turns. See
        /// <see cref="SpeculativeExecution.ResetStatsAndGovernor"/>.
        /// </summary>
        public void ResetStatsAndGovernor() => _exec.ResetStatsAndGovernor();

        /// <summary>
        /// Prefill the prompt through the trunk (capturing per-row hidden states
        /// when the speculator needs them) and hand the chunks to the speculator
        /// so its own state covers the whole prompt.
        /// Returns the last-position logits (vocab floats, caller-owned copy).
        /// </summary>
        public float[] Prefill(int[] promptTokens)
        {
            if (promptTokens == null || promptTokens.Length == 0)
                throw new ArgumentException("Prompt must not be empty.", nameof(promptTokens));

            // A trunk that keeps its draft head in sync internally wants the whole
            // prompt in one call: chunking it only adds a host round trip per
            // chunk, which stalls the trunk's own micro-batch pipelining.
            int chunkSize = _exec.PrefillSelfCatchUp
                ? Math.Max(1, promptTokens.Length)
                : Math.Max(1, PrefillChunkSize);
            float[] logits = null;
            int offset = 0;
            while (offset < promptTokens.Length)
            {
                int n = Math.Min(chunkSize, promptTokens.Length - offset);
                int[] chunk = new int[n];
                Array.Copy(promptTokens, offset, chunk, 0, n);
                // Linear trunk: the chunk's start position is the model's
                // current cache length (contiguous from the ResetKVCache).
                logits = _exec.PrefillStep(chunk, _model.CacheSeqLen);
                offset += n;
            }
            CarryPosition = _model.CacheSeqLen;
            return logits;
        }

        /// <summary>
        /// Greedy generation with speculative decoding. Resets the model KV
        /// cache, prefills <paramref name="promptTokens"/>, then emits up to
        /// <paramref name="maxNewTokens"/> tokens (the stop token, when hit, is
        /// included as the final element). <paramref name="onToken"/>, when
        /// given, receives every emitted token as it is produced and returns
        /// false to stop generating.
        /// </summary>
        public List<int> GenerateGreedy(int[] promptTokens, int maxNewTokens, Func<int, bool> isStopToken = null,
            Func<int, bool> onToken = null)
        {
            _model.ResetKVCache();
            _exec.Reset();

            var output = new List<int>();
            if (maxNewTokens <= 0)
                return output;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] logits = Prefill(promptTokens);
            LastPrefillSeconds = sw.Elapsed.TotalSeconds;
            GenerateFrom(logits, promptTokens.Length, maxNewTokens, null, isStopToken, onToken, output);
            return output;
        }

        /// <summary>
        /// Greedy speculative generation continuing from <paramref name="promptLogits"/>,
        /// the next-token logits at trunk position <paramref name="position"/>. Unlike
        /// <see cref="GenerateGreedy"/> this touches neither the KV cache nor the
        /// speculative state, so a caller that owns prompt prefill (a chat turn reusing
        /// the prefix its predecessor left in the cache) can drive the same draft/verify
        /// loop. <paramref name="onToken"/> receives each token as it is emitted and
        /// returns false to stop; the stop token is appended to the result but is not
        /// streamed.
        ///
        /// On return the trunk holds exactly the first
        /// <c>CacheSeqLen - position</c> tokens of the result, so a caller tracking the
        /// cache can mirror it without guessing where a stopped step left off.
        /// </summary>
        public List<int> GenerateGreedyFrom(float[] promptLogits, int position, int maxNewTokens,
            Func<int, bool> isStopToken = null, Func<int, bool> onToken = null)
        {
            if (promptLogits == null)
                throw new ArgumentNullException(nameof(promptLogits));

            var output = new List<int>();
            if (maxNewTokens > 0)
                GenerateFrom(promptLogits, position, maxNewTokens, null, isStopToken, onToken, output);
            return output;
        }

        /// <summary>
        /// Speculative generation that draws every emitted token with
        /// <paramref name="sampler"/> instead of argmax. Resets the model KV cache,
        /// prefills <paramref name="promptTokens"/>, then decodes as
        /// <see cref="GenerateSampledFrom"/> does.
        /// </summary>
        public List<int> GenerateSampled(int[] promptTokens, int maxNewTokens, TokenSampler sampler,
            Func<int, bool> isStopToken = null, Func<int, bool> onToken = null)
        {
            ArgumentNullException.ThrowIfNull(sampler);

            _model.ResetKVCache();
            _exec.Reset();

            var output = new List<int>();
            if (maxNewTokens <= 0)
                return output;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] logits = Prefill(promptTokens);
            LastPrefillSeconds = sw.Elapsed.TotalSeconds;
            GenerateFrom(logits, promptTokens.Length, maxNewTokens, sampler, isStopToken, onToken, output);
            return output;
        }

        /// <summary>
        /// Speculative generation continuing from <paramref name="promptLogits"/> that
        /// draws every emitted token with <paramref name="sampler"/> — the sampled
        /// counterpart of <see cref="GenerateGreedyFrom"/>, and the same contract the
        /// engine's <c>BatchExecutor</c> gives a served request.
        ///
        /// Speculation stays distribution-faithful because every emitted token is
        /// drawn by <paramref name="sampler"/> from a TRUNK row: the draft only
        /// decides whether a row was already computed. Three details make that true
        /// rather than approximately true, and each is a silent bias if dropped:
        ///
        ///   * the token drawn while verifying a row is the one emitted — re-drawing
        ///     it later from the same logits would bias the stream toward the drafts;
        ///   * accepted drafts enter the penalty history BEFORE the next row is
        ///     drawn, so a penalized sampler sees exactly the history plain decoding
        ///     would have given it;
        ///   * the draft head's logits are penalized with that same history (plus the
        ///     drafts pending in the window), so drafting argmaxes the distribution
        ///     verification draws from. Without it acceptance decays toward zero as
        ///     the history grows.
        ///
        /// The caller's own token bookkeeping stays in <paramref name="onToken"/>
        /// order; the penalty history kept here is private to the sampler.
        /// </summary>
        public List<int> GenerateSampledFrom(float[] promptLogits, int position, int maxNewTokens,
            TokenSampler sampler, Func<int, bool> isStopToken = null, Func<int, bool> onToken = null)
        {
            if (promptLogits == null)
                throw new ArgumentNullException(nameof(promptLogits));
            ArgumentNullException.ThrowIfNull(sampler);

            var output = new List<int>();
            if (maxNewTokens > 0)
                GenerateFrom(promptLogits, position, maxNewTokens, sampler, isStopToken, onToken, output);
            return output;
        }

        /// <param name="sampler">Null for argmax verification, otherwise the sampler
        /// every emitted token is drawn with.</param>
        private void GenerateFrom(float[] promptLogits, int position, int maxNewTokens,
            TokenSampler sampler, Func<int, bool> isStopToken, Func<int, bool> onToken, List<int> output)
        {
            // Standalone callers can supply an unbound sampler, or no sampler for
            // greedy generation. Model exclusions apply to both, as in the engine.
            var suppressed = _model.Tokenizer?.SuppressedTokenIds;
            bool hasSuppression = suppressed is { Count: > 0 };
            void Mask(float[] logits)
            {
                if (!hasSuppression) return;
                for (int i = 0; i < suppressed.Count; i++)
                    if ((uint)suppressed[i] < (uint)logits.Length)
                        logits[suppressed[i]] = float.NegativeInfinity;
            }
            bool IsSuppressed(int token)
            {
                if (hasSuppression)
                    for (int i = 0; i < suppressed.Count; i++)
                        if (suppressed[i] == token) return true;
                return false;
            }
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                Decode();

                // A verify batch commits the whole accepted prefix at once, so a run
                // that ends part-way through emitting one (a stop token, a consumer
                // that asked to stop) leaves the trunk holding tokens that never
                // reached the result. Drop them, so the trunk always holds exactly a
                // prefix of what was returned and a caller mirroring the cache does
                // not have to reason about where the last step stopped.
                int emitted = position + output.Count;
                if (_model.CacheSeqLen > emitted)
                {
                    // Rewinding drops trunk rows the carry-in hidden state was taken
                    // from, so it no longer describes the token the trunk ends on.
                    _model.SpecRewindCache(emitted);
                    CarryPosition = -1;
                }
                else
                {
                    // The trunk's own length, NOT `emitted`: decoding forwards every
                    // emitted token except the last, which is drawn from the previous
                    // row's logits and fed on the next step. The carry-in describes
                    // the token the trunk actually ends on.
                    CarryPosition = _model.CacheSeqLen;
                }
            }
            finally
            {
                LastDecodeSeconds = sw.Elapsed.TotalSeconds;
            }

            void Decode()
            {
                // Penalty history for a sampled run: exactly the tokens drawn so
                // far. Kept apart from `output` because the two move at different
                // times — an accepted draft has to be in the history BEFORE the
                // next verify row is drawn (mid-DecodeStep), while it reaches
                // `output` only once the step returns. Unused under argmax, where
                // no history is consulted.
                List<int> history = sampler != null ? _historyScratch : null;
                history?.Clear();

                // Draws the token a trunk row emits, and records it in the penalty
                // history. Accepted drafts go through _exec's onDraftAccepted hook
                // instead — they were drawn by the DRAFT head, not from this row.
                int Draw(float[] rowLogits)
                {
                    Mask(rowLogits);
                    int token = sampler == null ? Argmax(rowLogits, _vocab) : sampler.Sample(rowLogits, history);
                    if (IsSuppressed(token) || (hasSuppression && float.IsNegativeInfinity(rowLogits[token])))
                        throw new InvalidOperationException("No permitted generation token was selected; model exclusions cannot be overridden by a forced token or other sampler constraint.");
                    return token;
                }

                // Appends a token to the result and reports whether generation
                // should continue: a stop token ends the run without being
                // streamed, and the consumer can end it too (cancellation, a
                // stop sequence).
                bool Emit(int t)
                {
                    output.Add(t);
                    if (isStopToken != null && isStopToken(t))
                        return false;
                    return onToken == null || onToken(t);
                }

                int tLast = Draw(promptLogits);
                history?.Add(tLast);
                if (!Emit(tLast))
                    return;

                int n = position;

                while (output.Count < maxNewTokens)
                {
                    // Never draft more than we can still emit; the core also
                    // clamps to MaxDraftTokens and the context headroom.
                    int kMax = maxNewTokens - output.Count;

                    _acceptedScratch.Clear();
                    SpeculativeStepOutcome outcome = _exec.DecodeStep(
                        tLast, n, kMax,
                        drawNext: Draw,
                        // Penalty-aligned drafting: the draft head must argmax the
                        // same penalized distribution verification draws from, or
                        // acceptance decays toward zero as the history grows.
                        adjustDraftLogits: sampler == null && !hasSuppression
                            ? null
                            : (draftLogits, pendingDrafts) =>
                            {
                                Mask(draftLogits);
                                sampler?.ApplyPenalties(draftLogits, history, pendingDrafts);
                            },
                        onDraftAccepted: d =>
                        {
                            history?.Add(d);
                            _acceptedScratch.Add(d);
                        });

                    if (!outcome.UsedSpeculation)
                    {
                        // Plain decode step.
                        int next = Draw(outcome.NextLogits);
                        history?.Add(next);
                        n += 1;
                        tLast = next;
                        if (!Emit(next))
                            return;
                        continue;
                    }

                    // The trunk has already committed the accepted prefix plus the
                    // corrected/bonus token, so advance the position before emitting:
                    // a consumer that stops mid-step must still see a position that
                    // matches the cache.
                    n += outcome.AcceptedCount + 1;
                    tLast = outcome.NextToken;
                    // The token drawn from the first mismatching (or bonus) row is
                    // the sequence's next output token. It is emitted as drawn —
                    // re-drawing it from NextLogits would bias the stream toward
                    // the drafts — and enters the history here because the accept
                    // loop only reports the drafts it accepted.
                    history?.Add(outcome.NextToken);

                    // Emit accepted drafts then the corrected/bonus token.
                    int emittedThisStep = 0;
                    for (int i = 0; i < _acceptedScratch.Count && output.Count < maxNewTokens; i++)
                    {
                        emittedThisStep++;
                        if (!Emit(_acceptedScratch[i]))
                            return;
                    }
                    if (output.Count < maxNewTokens)
                    {
                        emittedThisStep++;
                        if (!Emit(outcome.NextToken))
                            return;
                    }

                    if (emittedThisStep == 0)
                        break; // budget exhausted mid-step
                }
            }
        }

        private static int Argmax(float[] logits, int vocab)
        {
            int best = 0;
            float bestVal = logits[0];
            for (int i = 1; i < vocab; i++)
            {
                float v = logits[i];
                if (v > bestVal)
                {
                    bestVal = v;
                    best = i;
                }
            }
            return best;
        }
    }
}
