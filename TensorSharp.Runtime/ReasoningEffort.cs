// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// The OpenAI <c>reasoning_effort</c> levels, as a chat template reads them.
    ///
    /// <para>
    /// Only Harmony (GPT-OSS) renders the level today: its system message carries a
    /// <c>Reasoning: low|medium|high</c> line the model was trained on, and it is the
    /// ONLY lever over that family's chain of thought. GPT-OSS cannot be asked not to
    /// reason - it always opens the <c>analysis</c> channel before its answer - so a
    /// request that explicitly turns thinking off and names no effort is rendered at
    /// <see cref="Low"/> instead (see <see cref="ForRequest"/>). Families without such
    /// a line ignore the value.
    /// </para>
    /// </summary>
    public static class ReasoningEffort
    {
        public const string Low = "low";
        public const string Medium = "medium";
        public const string High = "high";

        /// <summary>The level a request that names none renders at.</summary>
        public const string Default = Medium;

        public static readonly string[] Levels = { Low, Medium, High };

        /// <summary>
        /// Canonical spelling of a level in <paramref name="level"/>, or false when
        /// <paramref name="value"/> is not one. Null and empty input are "not given"
        /// (true, level null), never an error.
        /// </summary>
        public static bool TryNormalize(string? value, out string? level)
        {
            level = null;
            if (string.IsNullOrWhiteSpace(value))
                return true;
            foreach (string known in Levels)
            {
                if (string.Equals(known, value.Trim(), StringComparison.OrdinalIgnoreCase))
                {
                    level = known;
                    return true;
                }
            }
            return false;
        }

        /// <summary>The level to render, defaulting an absent or unknown one.</summary>
        public static string Resolve(string? level)
            => TryNormalize(level, out string? known) && known != null ? known : Default;

        /// <summary>
        /// The level a request runs at: the one it named, else <see cref="Low"/> when it
        /// EXPLICITLY turned thinking off (<c>"think": false</c> in the body, not merely
        /// absent), else null for the family's default. A 256-token GPT-OSS request with
        /// thinking off used to end with <c>finish_reason=length</c> and no content
        /// because the prompt still said <c>Reasoning: medium</c>.
        /// </summary>
        public static string? ForRequest(string? requested, bool thinkExplicitlyFalse)
        {
            TryNormalize(requested, out string? level);
            return level ?? (thinkExplicitlyFalse ? Low : null);
        }
    }
}
