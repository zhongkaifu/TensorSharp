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
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// Greedy speculative decoding driven by a Qwen3.5/3.6 NextN/MTP draft head
    /// (llama.cpp's `--spec-type draft-mtp`, vLLM's qwen3_5_mtp speculator).
    ///
    /// Thin standalone wrapper over <see cref="MtpSpeculativeExecution"/> (the
    /// shared draft/verify/rollback/catch-up core also driven by the engine's
    /// <c>BatchExecutor</c>): owns the whole generate loop — KV reset, chunked
    /// prompt prefill, greedy argmax verification — for tests and offline use.
    ///
    /// Greedy verification makes the output token stream identical to plain
    /// greedy decoding up to batched-vs-sequential floating point drift.
    /// Single-sequence; the caller owns the model's KV cache lifecycle.
    /// </summary>
    public sealed class MtpSpeculativeDecoder
    {
        private readonly Qwen35Model _model;
        private readonly MtpSpeculativeExecution _exec;
        private readonly int _vocab;
        private readonly List<int> _acceptedScratch = new();

        /// <summary>Maximum tokens drafted per speculative step (llama.cpp n_max).</summary>
        public int MaxDraftTokens => _exec.MaxDraftTokens;

        /// <summary>
        /// Minimum draft confidence (top-1 probability over the draft head's
        /// top-10 logits, matching llama.cpp's top-k(10) draft sampler) for a
        /// drafted token to be kept. Drafting stops at the first low-confidence
        /// token.
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
        public double AcceptanceRate => _exec.Stats.AcceptanceRate;

        /// <summary>Wall-clock phase breakdown of the speculative decode loop
        /// (draft / verify / snapshot / rollback / catch-up / plain).</summary>
        public MtpSpecStats Stats => _exec.Stats;

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent in prompt prefill.</summary>
        public double LastPrefillSeconds { get; private set; }

        /// <summary>Wall-clock seconds the last GenerateGreedy call spent past prefill.</summary>
        public double LastDecodeSeconds { get; private set; }

        // Default draft window: 8. The MinDraftProb gate stops drafting at the
        // first low-confidence token, so a longer window only extends confident
        // streaks — measured 1.21x vs 1.08x (window 4) on Qwen3.6-35B-A3B
        // ggml_cpu at unchanged 86% acceptance; neutral on 27B ggml_cuda.
        public MtpSpeculativeDecoder(Qwen35Model model, int maxDraftTokens = 8)
        {
            _model = model ?? throw new ArgumentNullException(nameof(model));
            _exec = new MtpSpeculativeExecution(model, maxDraftTokens);
            _vocab = model.Config.VocabSize;
        }

        /// <summary>Reset speculative state and statistics. Does NOT touch the model's KV cache.</summary>
        public void Reset() => _exec.Reset();

        /// <summary>
        /// Prefill the prompt through the trunk (capturing h_nextn) and replay it
        /// through the MTP block so its KV cache covers the whole prompt.
        /// Returns the last-position logits (vocab floats, caller-owned copy).
        /// </summary>
        public float[] Prefill(int[] promptTokens)
        {
            if (promptTokens == null || promptTokens.Length == 0)
                throw new ArgumentException("Prompt must not be empty.", nameof(promptTokens));

            int chunkSize = Math.Max(1, PrefillChunkSize);
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
            return logits;
        }

        /// <summary>
        /// Greedy generation with MTP speculative decoding. Resets the model KV
        /// cache, prefills <paramref name="promptTokens"/>, then emits up to
        /// <paramref name="maxNewTokens"/> tokens (the stop token, when hit, is
        /// included as the final element).
        /// </summary>
        public List<int> GenerateGreedy(int[] promptTokens, int maxNewTokens, Func<int, bool> isStopToken = null)
        {
            _model.ResetKVCache();
            _exec.Reset();

            var output = new List<int>();
            if (maxNewTokens <= 0)
                return output;

            var sw = System.Diagnostics.Stopwatch.StartNew();
            float[] logits = Prefill(promptTokens);
            LastPrefillSeconds = sw.Elapsed.TotalSeconds;
            sw.Restart();
            try
            {
                int tLast = Argmax(logits, _vocab);
                output.Add(tLast);
                if (isStopToken != null && isStopToken(tLast))
                    return output;

                int n = promptTokens.Length;

                while (output.Count < maxNewTokens)
                {
                    // Never draft more than we can still emit; the core also
                    // clamps to MaxDraftTokens and the context headroom.
                    int kMax = maxNewTokens - output.Count;

                    _acceptedScratch.Clear();
                    MtpDecodeOutcome outcome = _exec.DecodeStep(
                        tLast, n, kMax,
                        drawNext: row => Argmax(row, _vocab),
                        adjustDraftLogits: null,
                        onDraftAccepted: _acceptedScratch.Add);

                    if (!outcome.UsedSpeculation)
                    {
                        // Plain decode step.
                        int next = Argmax(outcome.NextLogits, _vocab);
                        output.Add(next);
                        n += 1;
                        tLast = next;
                        if (isStopToken != null && isStopToken(next))
                            return output;
                        continue;
                    }

                    // Emit accepted drafts then the corrected/bonus token.
                    int emittedThisStep = 0;
                    for (int i = 0; i < _acceptedScratch.Count && output.Count < maxNewTokens; i++)
                    {
                        output.Add(_acceptedScratch[i]);
                        emittedThisStep++;
                        if (isStopToken != null && isStopToken(_acceptedScratch[i]))
                            return output;
                    }
                    if (output.Count < maxNewTokens)
                    {
                        output.Add(outcome.NextToken);
                        emittedThisStep++;
                        if (isStopToken != null && isStopToken(outcome.NextToken))
                            return output;
                    }

                    n += outcome.AcceptedCount + 1;
                    tLast = outcome.NextToken;
                    if (emittedThisStep == 0)
                        break; // budget exhausted mid-step
                }

                return output;
            }
            finally
            {
                LastDecodeSeconds = sw.Elapsed.TotalSeconds;
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
