// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System.Collections.Generic;

namespace InferenceEngine
{
    /// <summary>
    /// Configuration for token sampling during inference.
    /// Default values produce greedy (deterministic) decoding.
    /// </summary>
    public class SamplingConfig
    {
        /// <summary>
        /// Controls randomness. 0 = greedy/deterministic, higher = more random.
        /// Typical range: 0.0 - 2.0.
        /// </summary>
        public float Temperature { get; set; } = 0f;

        /// <summary>
        /// Limits sampling to the top K most probable tokens. 0 = disabled.
        /// </summary>
        public int TopK { get; set; } = 0;

        /// <summary>
        /// Nucleus sampling: limits sampling to the smallest set of tokens
        /// whose cumulative probability exceeds this value. 1.0 = disabled.
        /// Typical range: 0.0 - 1.0.
        /// </summary>
        public float TopP { get; set; } = 1.0f;

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
        /// Typical range: 1.0 - 2.0.
        /// </summary>
        public float RepetitionPenalty { get; set; } = 1.0f;

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
        /// Random seed for reproducible sampling. -1 = non-deterministic (time-based seed).
        /// </summary>
        public int Seed { get; set; } = -1;

        /// <summary>
        /// Stop sequences: generation stops when any of these strings is produced.
        /// The stop string itself is not included in the output.
        /// </summary>
        public List<string> StopSequences { get; set; }

        /// <summary>
        /// Maximum number of tokens to generate. 0 = use caller's default.
        /// </summary>
        public int MaxTokens { get; set; } = 0;

        /// <summary>
        /// Returns true if this config is effectively greedy decoding.
        /// </summary>
        public bool IsGreedy => Temperature <= 0f && TopK <= 0 && TopP >= 1.0f && MinP <= 0f;

        /// <summary>
        /// Default config: greedy decoding.
        /// </summary>
        public static SamplingConfig Greedy => new SamplingConfig();

        /// <summary>
        /// Sensible creative defaults (temperature=0.7, top_p=0.9, min_p=0.05).
        /// </summary>
        public static SamplingConfig Creative => new SamplingConfig
        {
            Temperature = 0.7f,
            TopP = 0.9f,
            MinP = 0.05f,
        };
    }
}
