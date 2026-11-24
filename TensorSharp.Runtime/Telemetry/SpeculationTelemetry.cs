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
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Runtime.Telemetry
{
    /// <summary>
    /// <c>inference.speculation</c>: one span for a request that finished while speculation was on, a
    /// child of the request's own span, from its first token to the moment the engine finished it. One
    /// that speculated (<c>tensorsharp.speculative.outcome</c> <c>ran</c>) carries what
    /// <see cref="SpeculationStats"/> counted; one that did not (<c>declined</c>) carries why
    /// (<c>tensorsharp.speculative.declined</c>, a <see cref="SpeculationDeclineReason"/>). The engine
    /// opens it on its worker thread as the request finishes and writes its "Speculative decoding stats"
    /// line while the span is current, so the line lands in the request's trace rather than in none.
    /// </summary>
    public static class SpeculationTelemetry
    {
        public const string SpanName = "inference.speculation";

        // Span-only attributes: the per-request totals the stats line prints. Per verify step values
        // are a phase over tensorsharp.speculative.steps.verify, so they are left to the query.
        public const string TokensDrafted = "tensorsharp.speculative.tokens.drafted";
        public const string TokensAccepted = "tensorsharp.speculative.tokens.accepted";
        /// <summary>Accepted over drafted, 0 to 1.</summary>
        public const string Acceptance = "tensorsharp.speculative.acceptance";
        /// <summary>The most tokens one verify window drafts.</summary>
        public const string DraftWindow = "tensorsharp.speculative.draft_window";
        public const string VerifySteps = "tensorsharp.speculative.steps.verify";
        public const string PlainSteps = "tensorsharp.speculative.steps.plain";
        /// <summary>Verify steps that kept fewer tokens than they drafted.</summary>
        public const string RollbackSteps = "tensorsharp.speculative.steps.rollback";
        /// <summary>Steps the cost governor kept plain because drafting measured slower.</summary>
        public const string ParkedSteps = "tensorsharp.speculative.steps.parked";
        public const string DraftMs = "tensorsharp.speculative.phase.draft_ms";
        public const string VerifyMs = "tensorsharp.speculative.phase.verify_ms";
        public const string CommitMs = "tensorsharp.speculative.phase.commit_ms";
        public const string SnapshotMs = "tensorsharp.speculative.phase.snapshot_ms";
        public const string RollbackMs = "tensorsharp.speculative.phase.rollback_ms";
        public const string CatchUpMs = "tensorsharp.speculative.phase.catch_up_ms";
        public const string PlainMs = "tensorsharp.speculative.phase.plain_ms";
        /// <summary>The cost governor's measured ms per emitted token, plain and speculative (0 before it has samples).</summary>
        public const string GovernorPlainMsPerToken = "tensorsharp.speculative.governor.plain_ms_per_token";
        public const string GovernorSpecMsPerToken = "tensorsharp.speculative.governor.spec_ms_per_token";
        public const string GovernorWins = "tensorsharp.speculative.governor.wins";
        public const string GovernorLosses = "tensorsharp.speculative.governor.losses";

        /// <summary>
        /// Start the span of a request whose engine work is <paramref name="stats"/>, under
        /// <paramref name="request"/> (the request span's context, <c>SequenceState.TelemetryContext</c>),
        /// from <paramref name="decodeStartUtc"/> (its first token; now when unknown). Null when the request
        /// has no span (telemetry off, or a caller that traces nothing), so it never becomes a root of its
        /// own trace. Stop it (dispose) where the request ends.
        /// </summary>
        public static Activity? Start(ActivityContext request, DateTime? decodeStartUtc, SpeculationStats stats)
        {
            ArgumentNullException.ThrowIfNull(stats);
            var span = StartSpan(request, decodeStartUtc);
            if (span is { IsAllDataRequested: true })
                SetTags(span, stats);
            return span;
        }

        /// <summary>
        /// Start the span of a request that finished without speculating although speculation was on,
        /// with <paramref name="reason"/> (a <see cref="SpeculationDeclineReason"/>). Null as for
        /// <see cref="Start(ActivityContext, DateTime?, SpeculationStats)"/>.
        /// </summary>
        public static Activity? StartDeclined(ActivityContext request, DateTime? decodeStartUtc, string reason)
        {
            ArgumentException.ThrowIfNullOrEmpty(reason);
            var span = StartSpan(request, decodeStartUtc);
            if (span is { IsAllDataRequested: true })
            {
                span.SetTag(TelemetryAttributes.SpeculativeOutcome, TelemetryAttributes.SpeculativeOutcomeDeclined);
                span.SetTag(TelemetryAttributes.SpeculativeDeclined, reason);
            }
            return span;
        }

        /// <summary><c>tensorsharp.speculative.requests</c>: one request, by outcome, with its speculator
        /// (ran) or reason (declined).</summary>
        public static void CountRequest(string architecture, string? speculator, string? declined)
        {
            var counter = TensorSharpMetrics.SpeculativeRequests;
            if (!counter.Enabled)
                return;
            var architectureTag = new System.Collections.Generic.KeyValuePair<string, object?>(
                TelemetryAttributes.ModelArchitecture, string.IsNullOrEmpty(architecture) ? "unknown" : architecture);
            if (declined != null)
                counter.Add(1, architectureTag,
                    new(TelemetryAttributes.SpeculativeOutcome, TelemetryAttributes.SpeculativeOutcomeDeclined),
                    new(TelemetryAttributes.SpeculativeDeclined, declined));
            else
                counter.Add(1, architectureTag,
                    new(TelemetryAttributes.SpeculativeOutcome, TelemetryAttributes.SpeculativeOutcomeRan),
                    new(TelemetryAttributes.Speculator, speculator ?? "unknown"));
        }

        private static Activity? StartSpan(ActivityContext request, DateTime? decodeStartUtc)
        {
            if (request == default || !TensorSharpTelemetry.Inference.HasListeners())
                return null;
            return TensorSharpTelemetry.Inference.StartActivity(SpanName, ActivityKind.Internal, request,
                startTime: decodeStartUtc is { } start ? TensorSharpTelemetry.ToUtcOffset(start) : default);
        }

        private static void SetTags(Activity span, SpeculationStats stats)
        {
            span.SetTag(TelemetryAttributes.SpeculativeOutcome, TelemetryAttributes.SpeculativeOutcomeRan);
            if (!string.IsNullOrEmpty(stats.Speculator)) span.SetTag(TelemetryAttributes.Speculator, stats.Speculator);
            if (stats.DraftWindow > 0) span.SetTag(DraftWindow, stats.DraftWindow);
            span.SetTag(TokensDrafted, stats.TokensDrafted);
            span.SetTag(TokensAccepted, stats.TokensAccepted);
            span.SetTag(Acceptance, Math.Round(stats.AcceptanceRate, 4));
            span.SetTag(VerifySteps, stats.VerifySteps);
            span.SetTag(PlainSteps, stats.PlainSteps);
            span.SetTag(RollbackSteps, stats.RollbackSteps);
            span.SetTag(ParkedSteps, stats.GovernorParkedSteps);
            span.SetTag(DraftMs, Ms(stats.DraftMs));
            span.SetTag(VerifyMs, Ms(stats.VerifyMs));
            span.SetTag(CommitMs, Ms(stats.CommitMs));
            span.SetTag(SnapshotMs, Ms(stats.SnapshotMs));
            span.SetTag(RollbackMs, Ms(stats.RollbackMs));
            span.SetTag(CatchUpMs, Ms(stats.CatchUpMs));
            span.SetTag(PlainMs, Ms(stats.PlainMs));
            span.SetTag(GovernorPlainMsPerToken, Ms(stats.PlainMsPerToken));
            span.SetTag(GovernorSpecMsPerToken, Ms(stats.SpecMsPerToken));
            span.SetTag(GovernorWins, stats.GovernorWins);
            span.SetTag(GovernorLosses, stats.GovernorLosses);
        }

        private static double Ms(double ms) => Math.Round(ms, 1);
    }
}
