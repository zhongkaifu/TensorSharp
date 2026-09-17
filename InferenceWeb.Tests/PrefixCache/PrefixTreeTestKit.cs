// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>A block host with per-block flags and reference counts (tree refs + holder refs).</summary>
internal sealed class FakePageHost : IPrefixTreePageHost
{
    private readonly int _blockSize;
    private readonly Dictionary<KvBlock, (bool Paged, bool Snapshot, int Used)> _flags = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<KvBlock, int> _refs = new(ReferenceEqualityComparer.Instance);
    private int _nextId;

    internal FakePageHost(int blockSize) { _blockSize = blockSize; }

    internal long TreeRetains { get; private set; }
    internal long TreeFrees { get; private set; }
    internal long TreeRefs => TreeRetains - TreeFrees;

    /// <summary>A block holding one full page in the given store (owned by the caller: refcount 1).</summary>
    internal KvBlock NewBlock(PageStore store, bool fullA1 = true)
    {
        var b = new KvBlock(_nextId++);
        _flags[b] = ((store & PageStore.A2ModelPaged) != 0, (store & PageStore.A1HostSlab) != 0, fullA1 ? _blockSize : _blockSize / 2);
        _refs[b] = 1;
        return b;
    }

    /// <summary>A block with neither flag set (a page never written to a store of record).</summary>
    internal KvBlock NewUnbackedBlock()
    {
        var b = new KvBlock(_nextId++);
        _flags[b] = (false, false, 0);
        _refs[b] = 1;
        return b;
    }

    internal void Unset(KvBlock b, bool paged = false, bool snapshot = false)
    {
        var f = _flags[b];
        _flags[b] = (paged ? false : f.Paged, snapshot ? false : f.Snapshot, f.Used);
    }

    internal void DropCallerRef(KvBlock b) => _refs[b]--;

    public void RetainPage(KvBlock block) { _refs[block]++; TreeRetains++; }
    public void FreePage(KvBlock block) { _refs[block]--; TreeFrees++; if (_refs[block] < 0) throw new InvalidOperationException("block over-freed"); }
    public bool HoldsModelPagedKv(KvBlock block) => _flags[block].Paged;
    public bool HoldsSnapshotBytes(KvBlock block) => _flags[block].Snapshot;
    public int UsedTokens(KvBlock block) => _flags[block].Used;
    public int RefCount(KvBlock block) => _refs[block];
}

/// <summary>A validator whose refusals the test controls.</summary>
internal sealed class FakeValidator : IPayloadValidator
{
    internal readonly HashSet<string> Refused = new(StringComparer.Ordinal);
    internal Func<string, int, int, bool>? Rule;
    internal int Calls;

    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
    {
        Calls++;
        if (Refused.Contains(payloadKey)) return false;
        return Rule?.Invoke(payloadKey, payloadTokens, targetTokens) ?? true;
    }
}

internal sealed class FixedWaitingView : IWaitingPlanView
{
    internal bool Answer;
    internal int Calls;
    public bool NoOtherWaiterTargets(int scopeIx, RadixNode node, long treeVersion) { Calls++; return Answer; }
}

internal static class Tk
{
    internal const int B = 8;

    internal static PrefixCacheCapabilities Caps(
        EndStateSupport endState = EndStateSupport.CopyAndDonate,
        TruncationKind truncation = TruncationKind.Any,
        int truncationParameter = 0,
        int granularity = 1,
        int rewindCap = 16,
        PageSupport pages = PageSupport.Both,
        bool needStateAtEnd = false,
        int pageWindow = 0,
        bool copyPagedToHolder = false,
        bool primaryResident = true,
        bool adoptPrimary = true,
        bool reuseAcrossMedia = true,
        int mmReuseMin = 0,
        int minRetain = 1,
        int maxNativeSlots = 0,
        ResourceVector subCap = default) => new()
        {
            Class = FamilyClass.P,
            Readiness = PrefixCacheMode.Legacy,
            NamespaceFingerprint = "test-fp",
            EndState = endState,
            CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = adoptPrimary,
            PrimaryResident = primaryResident,
            MinRetainTokens = minRetain,
            Truncation = truncation,
            TruncationParameter = truncationParameter,
            TruncationGranularity = granularity,
            RewindCapTokens = rewindCap,
            Pages = pages,
            PagesNeedStateAtEnd = needStateAtEnd,
            PageWindowTokens = pageWindow,
            SupportsCopyPagedToHolder = copyPagedToHolder,
            ReuseAcrossMediaSpan = reuseAcrossMedia,
            MmReuseMinTokens = mmReuseMin,
            MaxRetainedNativeSlots = maxNativeSlots,
            SubCapBytes = subCap,
        };

    internal static PrefixTree Tree(
        PrefixCacheCapabilities? caps = null,
        IPrefixTreePageHost? host = null,
        IPayloadValidator? validator = null,
        int publicMax = 2,
        int scopedMax = 0,
        bool batchedPaged = true,
        int minClone = 0,
        long pageHostBytes = 100,
        Func<ResourceClass, long>? spare = null,
        ResourceVector optionCap = default,
        long poolPagesCap = 0,
        int contextLength = int.MaxValue,
        int truncationSearchNodes = 64,
        Func<long>? clock = null,
        bool branchSnapshots = false,
        bool blockedByScope = false,
        bool strictReceipts = true) =>
        new(new PrefixTreeOptions
        {
            Capabilities = caps ?? Caps(),
            BlockSize = B,
            PageHost = host ?? new FakePageHost(B),
            PayloadValidator = validator,
            PublicMax = publicMax,
            ScopedEndStateLeavesMax = scopedMax,
            BatchedPagedEnabled = batchedPaged,
            MinCloneTokens = minClone,
            PageHostBytes = pageHostBytes,
            QuerySpareBytes = spare,
            OptionCapBytes = optionCap,
            PoolPagesCap = poolPagesCap,
            ContextLength = contextLength,
            TruncationSearchNodes = truncationSearchNodes,
            ClockMs = clock ?? (() => 0),
            BranchSnapshots = branchSnapshots,
            ComputeBlockedByScope = blockedByScope,
            TrackReceipts = true,
            StrictReceipts = strictReceipts,
        });

    internal static int[] Seq(int start, int count) => Enumerable.Range(start, count).ToArray();

    internal static int[] Cat(params int[][] parts) => parts.SelectMany(p => p).ToArray();

    internal static KeyRope Key(PrefixTree t, int[] tokens, MediaSpanRecord[]? spans = null) =>
        RadixKeyBuilder.BuildPrompt(tokens, spans ?? Array.Empty<MediaSpanRecord>(), t.KeyPool);

    internal static int Scope(PrefixTree t, ScopeKind kind = ScopeKind.Session) => t.InternScope(ScopeId.NewFresh(), kind);

    internal static EndStatePayload Holder(PrefixTree t, long host = 1000, long device = 0, long state = 0, long native = 0,
                                           EndStateKind kind = EndStateKind.Holder, bool persisted = false, int tokens = 0) => new()
    {
        Key = t.MintKey(),
        Kind = kind,
        Origin = PayloadOrigin.Donation,
        Footprint = new PayloadFootprint(tokens, tokens, new ResourceVector { HostKv = host, DeviceKv = device, StateSnapshot = state, NativeSlot = native }, 0),
        Persisted = persisted,
    };

    internal static EndStatePayload Primary(PrefixTree t) => new()
    {
        Key = t.MintKey(),
        Kind = EndStateKind.PrimaryResident,
        Origin = PayloadOrigin.Donation,
    };

    /// <summary>Insert + attach a holder end state; returns the node.</summary>
    internal static RadixNode Put(PrefixTree t, KeyRope key, int len, int scopeIx, int p = 0, long host = 1000,
                                  NodeFlags flags = NodeFlags.None, MediaSpanRecord[]? spans = null, EndStatePayload? payload = null)
    {
        RadixNode n = t.Insert(key, len, scopeIx, p, flags, spans);
        Assert.Equal(len, n.Depth);
        AttachResult r = t.AttachEndState(n, payload ?? Holder(t, host));
        Assert.True(r is AttachResult.Attached or AttachResult.Revived, $"attach result {r}");
        return n;
    }

    internal static PageRef[] Pages(FakePageHost host, int count, PageStore store = PageStore.Both, int first = 0, bool stateAtEnd = true)
    {
        var pages = new PageRef[count];
        for (int i = 0; i < count; i++)
            pages[i] = new PageRef(host.NewBlock(store), first + i, store, stateAtEnd);
        return pages;
    }

    internal static MatchRequest Req(KeyRope key, int scopeIx, int p = 0, int? limit = null, MediaSpanRecord[]? spans = null,
                                     ExpectedRoute route = ExpectedRoute.PerSequenceFused, bool primaryAvailable = true, int? keyLength = null)
    {
        int len = keyLength ?? key.Length;
        return new MatchRequest(key, len, scopeIx, p, limit ?? len - 1, spans ?? Array.Empty<MediaSpanRecord>(), route, primaryAvailable);
    }

    internal static MatchPlan Plan(PrefixTree t, in MatchRequest r, IWaitingPlanView? waiting = null)
    {
        var plan = new MatchPlan();
        t.Plan(r, plan, waiting);
        return plan;
    }

    internal static void Valid(PrefixTree t, InvariantCheckContext ctx = default)
    {
        IReadOnlyList<InvariantViolation> v = PrefixTreeInvariantChecker.CheckOnce(t, ctx);
        Assert.True(v.Count == 0, "invariants violated:\n" + string.Join("\n", v));
    }

    internal static MediaSpanRecord Span(int start, int end, string contentId) => MediaSpanRecord.FromContentId(start, end, contentId);

    /// <summary>Two 64-hex content ids whose 63-bit key elements collide (same last word) but whose 256-bit ids differ.</summary>
    internal static (string A, string B) CollidingIds() =>
        ("1111111111111111" + "2222222222222222" + "3333333333333333" + "0123456789abcdef",
         "9999999999999999" + "2222222222222222" + "3333333333333333" + "0123456789abcdef");

    /// <summary>Token array with media placeholder ids (the model input) — keys replace them.</summary>
    internal static int[] WithPlaceholders(int length, params MediaSpanRecord[] spans)
    {
        int[] tokens = Seq(1, length);
        foreach (MediaSpanRecord s in spans)
            for (int i = s.Start; i < s.End; i++) tokens[i] = 999;
        return tokens;
    }
}
