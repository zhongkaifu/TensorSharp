// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// The option the image model's own Qwen3-VL-8B picked for a multiple-choice question
    /// (<see cref="QwenImageModel.ChooseAnswer"/>).
    /// </summary>
    /// <param name="Index">The chosen option, as an index into the options in the order given.</param>
    /// <param name="Probability">Its probability among the options, averaged over the option
    /// orders the question is asked in (one per rotation).</param>
    /// <param name="Margin">Its lead over the runner-up's averaged probability. Near zero means the
    /// model could not tell the two apart, which a caller can show rather than act on.</param>
    public sealed record ImageIntentChoice(int Index, float Probability, float Margin)
    {
        /// <summary>
        /// Every option's averaged probability, in the order the options were given; they sum to 1.
        /// A caller whose options group into fewer decisions (several pictures that could each be
        /// the one changed) sums these per group rather than reading <see cref="Margin"/>, which
        /// is the lead over the single runner-up.
        /// </summary>
        public IReadOnlyList<float> Probabilities { get; init; } = Array.Empty<float>();

        /// <summary>The longest of the scored prompts, in tokens: each order is one causal pass over its prompt.</summary>
        public int PromptTokens { get; init; }
    }
}
