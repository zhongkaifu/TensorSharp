// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>
/// Validators derived from <see cref="PrefixCacheCapabilities"/> (G-19, DESIGN §4.7, §5.3.1).
/// Evaluated once per engine, tested once with fakes; models never implement rules.
/// </summary>
internal sealed class ResumabilityRules
{
    internal ResumabilityRules(PrefixCacheCapabilities caps, int blockSize, bool batchedPagedEnabled)
    {
        Validate(caps, blockSize);
        Caps = caps;
        BlockSize = blockSize;
        BatchedPagedEnabled = batchedPagedEnabled;
    }

    internal PrefixCacheCapabilities Caps { get; }
    internal int BlockSize { get; }
    internal bool BatchedPagedEnabled { get; }

    /// <summary>Throws <see cref="ArgumentException"/> for a record the rules cannot evaluate.</summary>
    internal static void Validate(PrefixCacheCapabilities caps, int blockSize)
    {
        if (caps is null) throw new ArgumentNullException(nameof(caps));
        if (string.IsNullOrEmpty(caps.NamespaceFingerprint))
            throw new ArgumentException("PrefixCacheCapabilities.NamespaceFingerprint must be non-empty (K1a).", nameof(caps));
        if (blockSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockSize), blockSize, "The page size must be positive.");
        if (caps.TruncationGranularity < 1)
            throw new ArgumentException("TruncationGranularity must be at least 1.", nameof(caps));
        if (caps.RewindCapTokens < 0)
            throw new ArgumentException("RewindCapTokens must not be negative.", nameof(caps));
        if (caps.MinRetainTokens < 0 || caps.MmReuseMinTokens < 0 || caps.PageWindowTokens < 0 || caps.MaxRetainedNativeSlots < 0)
            throw new ArgumentException("Token and count capabilities must not be negative.", nameof(caps));
        if ((caps.Truncation == TruncationKind.WithinUnwrappedWindow || caps.Truncation == TruncationKind.WithinRingSlack)
            && caps.TruncationParameter <= 0)
            throw new ArgumentException($"Truncation {caps.Truncation} needs a positive TruncationParameter.", nameof(caps));
        if (caps.SubCapBytes.AnyNegative)
            throw new ArgumentException("SubCapBytes must not be negative.", nameof(caps));
    }

    /// <summary><c>Permitted(node, L) := node.ScopeIx == r.ScopeIx or (node.ScopeIx == 0 and L ≤ publicCap)</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static bool Permitted(int nodeScopeIx, int length, int requestScopeIx, int publicCap) =>
        nodeScopeIx == 0 ? length <= publicCap : nodeScopeIx == requestScopeIx;

    /// <summary>
    /// Clamp(x): the match limit (K6/K7), spans are atomic (M4), and no reuse past the first span
    /// start when the family cannot continue across media (M6, DEC-11).
    /// </summary>
    internal int ClampLength(int x, in MatchRequest r, ref ClampReasons reasons)
    {
        int limit = EffectiveMatchLimit(r);
        if (x > limit)
        {
            x = limit;
            reasons |= r.MatchLimit < r.KeyLength - 1 ? ClampReasons.Breakpoint : ClampReasons.LeaveOne;
        }
        MediaSpanRecord[] spans = r.Spans;
        if (spans is { Length: > 0 })
        {
            for (int i = 0; i < spans.Length; i++)
            {
                MediaSpanRecord s = spans[i];
                if (s.Start >= x) break;
                if (x < s.End)
                {
                    x = s.Start;
                    reasons |= ClampReasons.Media;
                    break;
                }
            }
            if (!Caps.ReuseAcrossMediaSpan && x > spans[0].Start)
            {
                x = spans[0].Start;
                reasons |= ClampReasons.MediaAcrossSpan;
            }
        }
        return Math.Max(0, x);
    }

    /// <summary>Clamp without recording reasons.</summary>
    internal int ClampLength(int x, in MatchRequest r)
    {
        ClampReasons ignored = ClampReasons.None;
        return ClampLength(x, r, ref ignored);
    }

    /// <summary>λ bounded by the key: at least one token stays un-matched (K6).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int EffectiveMatchLimit(in MatchRequest r) =>
        Math.Max(0, Math.Min(r.MatchLimit, Math.Min(r.KeyLength - 1, r.Key.Length)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int AlignDown(int x, int granularity) => granularity <= 1 ? x : x - (x % granularity);

    /// <summary>Whether a cached end state of <paramref name="cached"/> tokens may be truncated to <paramref name="target"/>.</summary>
    internal bool TruncationAllows(int cached, int target)
    {
        if (target > cached || target < 0) return false;
        return Caps.Truncation switch
        {
            TruncationKind.None => false,
            TruncationKind.Any => true,
            TruncationKind.WithinUnwrappedWindow => cached <= Caps.TruncationParameter,
            TruncationKind.WithinRingSlack => cached - target <= Caps.TruncationParameter,
            TruncationKind.ModelDecides => true,
            _ => false,
        };
    }

    /// <summary>The 16-token rewind cap (DEC-19); <see cref="TruncationKind.ModelDecides"/> has none.</summary>
    internal bool RewindWithinCap(int cached, int target) =>
        Caps.Truncation == TruncationKind.ModelDecides || cached - target <= Caps.RewindCapTokens;

    /// <summary>Whether a page ending at <paramref name="pageEnd"/> is inside the family's page window.</summary>
    internal bool PageInWindow(int pageEnd) => Caps.PageWindowTokens <= 0 || pageEnd <= Caps.PageWindowTokens;

    /// <summary>Whether the page's end is a resumable boundary (Nemotron needs recurrent state at the end).</summary>
    internal bool PageEndResumable(in PageRef page) => !Caps.PagesNeedStateAtEnd || page.StateAtEnd;

    /// <summary>
    /// RouteReads(store, mode) (§5.3.1): which materialization modes can read a page's store of record
    /// on the expected route.
    /// </summary>
    internal bool RouteCanRead(PageStore store, MaterializeMode mode, ExpectedRoute route)
    {
        bool a1 = (store & PageStore.A1HostSlab) != 0 && (Caps.Pages == PageSupport.A1HostSlab || Caps.Pages == PageSupport.Both);
        bool a2Store = (store & PageStore.A2ModelPaged) != 0;
        bool a2 = a2Store && (Caps.Pages == PageSupport.A2ModelPaged || Caps.Pages == PageSupport.Both);
        return mode switch
        {
            MaterializeMode.InjectA1Pages => a1,
            MaterializeMode.BindPagesInPlace => a2 && BatchedPagedEnabled,
            MaterializeMode.CopyA2PagesToHolder => a2Store && Caps.SupportsCopyPagedToHolder && route != ExpectedRoute.BatchedPaged,
            _ => false,
        };
    }

    /// <summary>
    /// Tie preference among page modes that reach the same length: the route's natural mode first
    /// (BatchedPaged → Bind; Primary and PerSequenceFused → Inject, then CopyA2 for A2-only pages).
    /// Lower is preferred.
    /// </summary>
    internal static int PageModePreference(MaterializeMode mode, ExpectedRoute route) => route switch
    {
        ExpectedRoute.BatchedPaged => mode switch
        {
            MaterializeMode.BindPagesInPlace => 0,
            MaterializeMode.InjectA1Pages => 1,
            _ => 2,
        },
        _ => mode switch
        {
            MaterializeMode.InjectA1Pages => 0,
            MaterializeMode.CopyA2PagesToHolder => 1,
            _ => 2,
        },
    };

    /// <summary>
    /// Tie order across candidates (§5.3.3): PrimaryResident &gt; EndState(donate) &gt; Pages(Bind) &gt;
    /// EndState(clone) &gt; Pages(Inject|CopyA2) &gt; TruncatedEndState(donate) &gt; TruncatedEndState(clone).
    /// Lower is preferred.
    /// </summary>
    internal static int TieRank(CandidateKind kind, MaterializeMode mode) => kind switch
    {
        CandidateKind.PrimaryResident => 0,
        CandidateKind.EndState => mode == MaterializeMode.DonateEndState ? 1 : 3,
        CandidateKind.Pages => mode == MaterializeMode.BindPagesInPlace ? 2 : 4,
        CandidateKind.TruncatedEndState => mode == MaterializeMode.DonateEndState ? 5 : 6,
        _ => 7,
    };

    /// <summary>Whether a mode makes a copy (subject to the DEC-13 cost rule).</summary>
    internal static bool IsClone(MaterializeMode mode) =>
        mode == MaterializeMode.CloneEndState || mode == MaterializeMode.ConvertPrimaryThenClone;
}
