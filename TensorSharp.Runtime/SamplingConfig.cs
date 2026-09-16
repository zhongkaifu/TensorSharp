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

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Configuration for token sampling during inference.
    /// Default values match Ollama defaults (temperature=0.8, top_k=40, top_p=0.9).
    /// </summary>
    public class SamplingConfig
    {
        /// <summary>
        /// Controls randomness. 0 = greedy/deterministic, higher = more random.
        /// Typical range: 0.0 - 2.0. Default matches Ollama (0.8).
        /// </summary>
        public float Temperature { get; set; } = 0.8f;

        /// <summary>
        /// Limits sampling to the top K most probable tokens. 0 = disabled.
        /// Default matches Ollama (40).
        /// </summary>
        public int TopK { get; set; } = 40;

        /// <summary>
        /// Nucleus sampling: limits sampling to the smallest set of tokens
        /// whose cumulative probability exceeds this value. 1.0 = disabled.
        /// Typical range: 0.0 - 1.0. Default matches Ollama (0.9).
        /// </summary>
        public float TopP { get; set; } = 0.9f;

        /// <summary>
        /// Minimum probability threshold. Tokens with probability below
        /// min_p * max_probability are excluded. 0.0 = disabled.
        /// Typical range: 0.0 - 1.0.
        /// </summary>
        public float MinP { get; set; } = 0f;

        /// <summary>
        /// Penalizes tokens that have appeared in the generated text.
        /// Applied multiplicatively to logits. 1.0 = no penalty.
        /// Values > 1.0 discourage repetition, &lt; 1.0 encourage it.
        /// Typical range: 1.0 - 2.0. Default matches Ollama (1.1).
        /// </summary>
        public float RepetitionPenalty { get; set; } = 1.1f;

        /// <summary>
        /// Number of most-recent generated tokens considered by repetition,
        /// presence, and frequency penalties. Default 64 matches llama.cpp and
        /// Ollama. 0 disables history penalties; -1 considers the full history.
        /// </summary>
        public int PenaltyLastN { get; set; } = 64;

        /// <summary>
        /// Additive penalty based on whether a token has appeared at all.
        /// 0.0 = disabled. Positive values discourage repeated topics.
        /// Typical range: 0.0 - 2.0.
        /// </summary>
        public float PresencePenalty { get; set; } = 0f;

        /// <summary>
        /// Additive penalty proportional to how many times a token has appeared.
        /// 0.0 = disabled. Positive values discourage word repetition.
        /// Typical range: 0.0 - 2.0.
        /// </summary>
        public float FrequencyPenalty { get; set; } = 0f;

        /// <summary>
        /// Let the engine's <c>RepetitionGuard</c> end THIS request when its output locks
        /// into a loop. Default true.
        ///
        /// <para>
        /// The host-wide switch is <c>SchedulerConfig.StopRepetition</c>; this is the
        /// per-request one, for the caller that legitimately asks for a long constant run
        /// — a zero-filled literal, a blank grid, N identical rows of test data — which is
        /// exactly periodic and would otherwise be stopped after 128 tokens. Both must be
        /// on for the guard to fire.
        /// </para>
        /// </summary>
        public bool StopRepetition { get; set; } = true;

        /// <summary>
        /// Random seed for reproducible sampling. -1 = non-deterministic (time-based seed).
        /// </summary>
        public int Seed { get; set; } = -1;

        /// <summary>
        /// Stop sequences: generation stops when any of these strings is produced.
        /// The stop string itself is not included in the output.
        /// </summary>
        public List<string>? StopSequences { get; set; }

        /// <summary>
        /// Maximum number of tokens to generate. 0 = use caller's default.
        /// </summary>
        public int MaxTokens { get; set; } = 0;

        /// <summary>
        /// Restricts the FIRST generated token of the request to these token ids
        /// (null/empty = unrestricted). Used by structured output
        /// (<c>response_format: json_object/json_schema</c>) to force the reply
        /// to open with the JSON object — the same effect llama.cpp achieves via
        /// its JSON grammar — so time-to-first-token reflects prefill latency
        /// instead of however much prose the model would have rambled before the
        /// first <c>{</c> (which the streaming JSON filter suppresses). The most
        /// probable allowed token is picked (deterministic); subsequent tokens
        /// sample normally.
        /// </summary>
        public IReadOnlyList<int>? FirstTokenAllowList { get; set; }

        /// <summary>
        /// Grammar constraint enforced at every decode step: tokens the grammar
        /// cannot accept are removed before any other sampling stage runs, so the
        /// output is structurally valid by construction rather than by inspection
        /// afterwards.
        /// </summary>
        /// <remarks>
        /// This is per-sequence mutable state, not a setting — it carries the
        /// live parser position — so a <see cref="Clone"/>d config deliberately
        /// does <b>not</b> copy it. Callers running more than one sequence must
        /// give each its own instance, otherwise two sequences would advance the
        /// same parser and constrain each other. See
        /// <c>TensorSharp.Runtime.Grammar.GrammarConstraint</c>.
        /// </remarks>
        public Grammar.GrammarConstraint? Grammar { get; set; }

        /// <summary>Optional immutable policy that closes an open reasoning
        /// channel at its budget and continues within MaxTokens.</summary>
        public ThinkingTokenBudget? ThinkingBudget { get; set; }

        /// <summary>
        /// The OpenAI <c>reasoning_effort</c> the request asked for (<c>low</c>,
        /// <c>medium</c>, <c>high</c>), or null for the family's default. It rides on
        /// the sampling config because that is the one per-request object every
        /// generation entry point already carries down to the prompt renderer; it is
        /// a prompt fact rather than a sampler one, and the renderer that reads it is
        /// <see cref="ChatTemplate.RenderHarmony"/>. See <see cref="Runtime.ReasoningEffort"/>.
        /// </summary>
        public string? ReasoningEffort { get; set; }

        /// <summary>
        /// Returns true if this config is effectively greedy decoding.
        /// </summary>
        public bool IsGreedy => Temperature <= 0f && TopK <= 0 && TopP >= 1.0f && MinP <= 0f;

        /// <summary>
        /// Default config: matches Ollama defaults (temperature=0.8, top_k=40, top_p=0.9).
        /// </summary>
        public static SamplingConfig Default => new SamplingConfig();

        /// <summary>
        /// Greedy (deterministic) decoding: always pick the most probable token.
        /// </summary>
        public static SamplingConfig Greedy => new SamplingConfig
        {
            Temperature = 0f,
            TopK = 0,
            TopP = 1.0f,
            RepetitionPenalty = 1.0f,
        };

        /// <summary>
        /// Sensible creative defaults (temperature=0.7, top_p=0.9, min_p=0.05).
        /// </summary>
        public static SamplingConfig Creative => new SamplingConfig
        {
            Temperature = 0.7f,
            TopP = 0.9f,
            MinP = 0.05f,
        };

        /// <summary>
        /// This config adjusted for a turn that is writing CODE, leaving every value
        /// somebody actually chose exactly as it is.
        ///
        /// <para>
        /// <b>The repetition penalty is the part that matters, and it is a correctness
        /// fix rather than a preference.</b> <see cref="RepetitionPenalty"/> 1.1 over a
        /// <see cref="PenaltyLastN"/> window of 64 tokens is two to four lines of Python:
        /// every fifth line of a loop body is penalised against the indentation, the
        /// <c>self.</c>, the <c>return</c> and the closing delimiters of the four above
        /// it. Code is legitimately repetitive in exactly the tokens that carry its
        /// structure, so penalising them is a mechanism for producing the malformed
        /// indentation and the drifted near-duplicate rewrites that the patcher and the
        /// rewrite watch exist to catch downstream. Neither reference implementation
        /// applies a repetition penalty to code: the Agents SDK's
        /// <c>ModelSettings.frequency_penalty</c> and <c>presence_penalty</c> both default
        /// to <c>None</c> and it sends nothing at all.
        /// </para>
        /// <para>
        /// The defaults being replaced are Ollama's chat defaults — <c>temperature=0.8,
        /// top_k=40, top_p=0.9</c> — inherited for API compatibility and never chosen for
        /// code. That is the whole justification: this is not a claim that 0.2 is the one
        /// true temperature, it is the removal of a chat-tuned default from a task that is
        /// not chat.
        /// </para>
        /// <para>
        /// <b>Only values still at their built-in default are touched.</b> A temperature
        /// that differs from <see cref="Default"/> was chosen by somebody — a client, an
        /// operator's flag, a config file — and a host-side preference must not overrule a
        /// deliberate request. This is the same rule the operator's own pinning logic
        /// states: anything left at the built-in default "is never fighting an operator
        /// decision".
        /// </para>
        /// </summary>
        /// <param name="temperature">
        /// The temperature to use for code, or null to leave temperature alone and adjust
        /// only the penalty.
        /// </param>
        public SamplingConfig ForCodingTurn(float? temperature)
        {
            var builtIn = new SamplingConfig();
            SamplingConfig adjusted = Clone();

            if (temperature is { } wanted
                && Math.Abs(Temperature - builtIn.Temperature) < 0.0001f)
            {
                adjusted.Temperature = wanted;
            }

            if (Math.Abs(RepetitionPenalty - builtIn.RepetitionPenalty) < 0.0001f)
                adjusted.RepetitionPenalty = 1.0f;

            return adjusted;
        }

        /// <summary>
        /// Returns a deep copy of this config. Useful when callers want to seed
        /// per-request defaults from a shared <see cref="SamplingConfig"/> and
        /// then override individual fields without mutating the shared instance.
        /// The <see cref="StopSequences"/> list is duplicated so adding entries
        /// to the clone does not bleed back into the source config.
        /// <para>
        /// <see cref="Grammar"/> is intentionally <b>not</b> copied: it holds a
        /// live parser position, so sharing one across cloned configs would make
        /// concurrent sequences advance each other's state. Each sequence must be
        /// given its own constraint after cloning.
        /// </para>
        /// </summary>
        public SamplingConfig Clone()
        {
            return new SamplingConfig
            {
                Temperature = Temperature,
                TopK = TopK,
                TopP = TopP,
                MinP = MinP,
                StopRepetition = StopRepetition,
                RepetitionPenalty = RepetitionPenalty,
                PenaltyLastN = PenaltyLastN,
                PresencePenalty = PresencePenalty,
                FrequencyPenalty = FrequencyPenalty,
                Seed = Seed,
                MaxTokens = MaxTokens,
                StopSequences = StopSequences != null ? new List<string>(StopSequences) : null,
                FirstTokenAllowList = FirstTokenAllowList,
                ThinkingBudget = ThinkingBudget,
                ReasoningEffort = ReasoningEffort,
            };
        }
    }
}
