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
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Which sampling parameters the operator pinned explicitly (via a
    /// <c>--config</c> file, a CLI flag, or a <c>TENSORSHARP_*</c> env var).
    /// Anything not listed here was left at the built-in default, so a client
    /// value for it is never fighting an operator decision.
    /// </summary>
    [Flags]
    public enum SamplingField
    {
        None = 0,
        Temperature = 1 << 0,
        TopK = 1 << 1,
        TopP = 1 << 2,
        MinP = 1 << 3,
        RepetitionPenalty = 1 << 4,
        PenaltyLastN = 1 << 5,
        PresencePenalty = 1 << 6,
        FrequencyPenalty = 1 << 7,
        Seed = 1 << 8,
        StopSequences = 1 << 9,
    }

    /// <summary>
    /// Who wins when a request carries a sampling parameter the operator also
    /// pinned on the server.
    /// </summary>
    public enum SamplingPrecedence
    {
        /// <summary>Server-pinned values win; unpinned parameters still come from the request.</summary>
        Config = 0,

        /// <summary>Request values always win; server values only fill in what the client omitted.</summary>
        Request = 1,
    }

    /// <summary>
    /// Server-wide sampling defaults plus the metadata needed to decide whether
    /// they outrank a client's request.
    ///
    /// The distinction that matters is "pinned" vs "left alone". Clients such as
    /// VS Code Copilot Chat hardcode <c>temperature</c>/<c>top_p</c> into every
    /// request body, so under the historical request-always-wins rule an
    /// operator's <c>"temperature": 0.6</c> could never take effect for them
    /// (github.com/zhongkaifu/TensorSharp/issues/113). Under the default
    /// <see cref="SamplingPrecedence.Config"/> mode a parameter the operator
    /// pinned is authoritative; every parameter they did not pin keeps the
    /// familiar request &gt; built-in-default behaviour. <c>--sampling-precedence
    /// request</c> restores the old rule for deployments that want per-request
    /// control (evals pinning <c>temperature: 0</c>, for example).
    ///
    /// Stop sequences are the one merge rather than a replace: a client's stop
    /// strings are how it terminates its own protocol (agent scaffolds rely on
    /// them), so under <see cref="SamplingPrecedence.Config"/> the pinned ones
    /// are added to the request's list instead of overwriting it.
    /// </summary>
    public sealed class SamplingDefaults
    {
        public SamplingDefaults(
            SamplingConfig values,
            SamplingField pinned = SamplingField.None,
            SamplingPrecedence precedence = SamplingPrecedence.Config)
        {
            Values = values ?? new SamplingConfig();
            Pinned = pinned;
            Precedence = precedence;
        }

        /// <summary>The resolved default values (CLI &gt; env &gt; built-in defaults). Never null.</summary>
        public SamplingConfig Values { get; }

        /// <summary>
        /// Treat a bare <see cref="SamplingConfig"/> as defaults with nothing
        /// pinned — i.e. plain fallback values that any request may override.
        /// Lets callers (and tests) that only care about the values keep passing
        /// a <see cref="SamplingConfig"/> to the request parsers.
        /// </summary>
        public static implicit operator SamplingDefaults(SamplingConfig values)
            => values == null ? null : new SamplingDefaults(values);

        /// <summary>Parameters the operator set explicitly.</summary>
        public SamplingField Pinned { get; }

        /// <summary>Whether pinned values outrank request values.</summary>
        public SamplingPrecedence Precedence { get; }

        /// <summary>True when at least one pinned parameter will override requests.</summary>
        public bool EnforcesAnyField =>
            Precedence == SamplingPrecedence.Config && Pinned != SamplingField.None;

        public bool IsPinned(SamplingField field) => (Pinned & field) != 0;

        /// <summary>
        /// Re-apply the operator's pinned values on top of a config that was
        /// already merged with the request body. A no-op under
        /// <see cref="SamplingPrecedence.Request"/> or when nothing was pinned,
        /// which keeps the historical behaviour byte-for-byte for servers
        /// started without any sampling flags.
        /// </summary>
        public SamplingConfig ApplyPinned(SamplingConfig parsed)
        {
            if (parsed == null || !EnforcesAnyField)
                return parsed;

            if (IsPinned(SamplingField.Temperature)) parsed.Temperature = Values.Temperature;
            if (IsPinned(SamplingField.TopK)) parsed.TopK = Values.TopK;
            if (IsPinned(SamplingField.TopP)) parsed.TopP = Values.TopP;
            if (IsPinned(SamplingField.MinP)) parsed.MinP = Values.MinP;
            if (IsPinned(SamplingField.RepetitionPenalty)) parsed.RepetitionPenalty = Values.RepetitionPenalty;
            if (IsPinned(SamplingField.PenaltyLastN)) parsed.PenaltyLastN = Values.PenaltyLastN;
            if (IsPinned(SamplingField.PresencePenalty)) parsed.PresencePenalty = Values.PresencePenalty;
            if (IsPinned(SamplingField.FrequencyPenalty)) parsed.FrequencyPenalty = Values.FrequencyPenalty;
            if (IsPinned(SamplingField.Seed)) parsed.Seed = Values.Seed;

            if (IsPinned(SamplingField.StopSequences) && Values.StopSequences != null)
            {
                // Union, not replace: the request's stop strings are load-bearing
                // for the client's own parsing, and an extra server-side stop can
                // only ever end generation earlier.
                var merged = parsed.StopSequences != null
                    ? new List<string>(parsed.StopSequences)
                    : new List<string>();
                foreach (string pinnedStop in Values.StopSequences)
                {
                    if (string.IsNullOrEmpty(pinnedStop)) continue;
                    if (!merged.Contains(pinnedStop, StringComparer.Ordinal))
                        merged.Add(pinnedStop);
                }
                parsed.StopSequences = merged;
            }

            return parsed;
        }

        /// <summary>Human-readable summary for the startup banner, e.g. <c>config(temperature,top-p)</c>.</summary>
        public string DescribePolicy()
        {
            if (Precedence == SamplingPrecedence.Request)
                return "request (client values win over server defaults)";
            if (Pinned == SamplingField.None)
                return "config (nothing pinned — every request keeps full control)";

            var names = new List<string>();
            if (IsPinned(SamplingField.Temperature)) names.Add("temperature");
            if (IsPinned(SamplingField.TopK)) names.Add("top-k");
            if (IsPinned(SamplingField.TopP)) names.Add("top-p");
            if (IsPinned(SamplingField.MinP)) names.Add("min-p");
            if (IsPinned(SamplingField.RepetitionPenalty)) names.Add("repeat-penalty");
            if (IsPinned(SamplingField.PenaltyLastN)) names.Add("repeat-last-n");
            if (IsPinned(SamplingField.PresencePenalty)) names.Add("presence-penalty");
            if (IsPinned(SamplingField.FrequencyPenalty)) names.Add("frequency-penalty");
            if (IsPinned(SamplingField.Seed)) names.Add("seed");
            if (IsPinned(SamplingField.StopSequences)) names.Add("stop(merged)");
            return "config (server-pinned: " + string.Join(",", names) + ")";
        }
    }
}
