// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// DESIGN §12.3 tree-level property harness. A seeded trace of chat-shaped operations (new chats,
/// follow-ups, regenerations, forks, replays, image turns, breakpoints, bursts, aborts, preemption,
/// publication, eviction, pressure, scope retirement, invalidation, backend recreation, faults) runs
/// against one <see cref="PrefixTree"/>. After every operation it checks the invariants, compares every
/// plan with a brute-force evaluation over all nodes, bounds reuse by what the scope could have computed
/// (isolation), and checks media-span atomicity, receipt balance and byte-accounting conservation.
/// </summary>
internal sealed class TreeTraceHarness
{
    internal const int BlockSize = 8;
    internal const int Steps = 300;
    private const int MinRetain = 4;
    private const int Placeholder = 999;
    private const long PageHostBytes = 16;

    private sealed class Profile
    {
        public required string Name;
        public required PrefixCacheCapabilities Caps;
        public bool BatchedPaged;
        public bool NativeSlots;
        public Func<string, int, int, bool>? ModelRule;
    }

    private sealed class Session
    {
        public ScopeId Id;
        public ScopeKind Kind;
        public int[] System = Array.Empty<int>();
        public List<int> Transcript = new();                 // tokens after the system prompt
        public List<MediaSpanRecord> Spans = new();          // absolute positions
    }

    private sealed class Request
    {
        public Session Session = null!;
        public ScopeId ScopeId;
        public int ScopeIx;
        public int P;
        public int[] Tokens = Array.Empty<int>();
        public MediaSpanRecord[] Spans = Array.Empty<MediaSpanRecord>();
        public KeyRope Key = null!;
        public int Limit;
        public int BreakpointLimit;                          // 0 = none
        public ExpectedRoute Route;
        public MatchPlan Plan = new();
        public LockReceipt Durable;
        public int Reused;
        public bool Published;
    }

    private sealed class RecordingHost : IPrefixTreePageHost
    {
        private readonly Dictionary<KvBlock, (bool Paged, bool Snapshot)> _flags = new(ReferenceEqualityComparer.Instance);
        private readonly Dictionary<KvBlock, int> _refs = new(ReferenceEqualityComparer.Instance);
        public readonly HashSet<KvBlock> TreeHeld = new(ReferenceEqualityComparer.Instance);
        private int _next;

        public KvBlock New(bool paged, bool snapshot)
        {
            var b = new KvBlock(_next++);
            _flags[b] = (paged, snapshot);
            _refs[b] = 1;
            return b;
        }

        public void DropCaller(KvBlock b) => _refs[b]--;
        public int TreeHeldSnapshots => TreeHeld.Count(b => _flags[b].Snapshot);
        public void RetainPage(KvBlock block) { _refs[block]++; if (!TreeHeld.Add(block)) throw new InvalidOperationException("tree retained a block twice"); }
        public void FreePage(KvBlock block) { _refs[block]--; if (!TreeHeld.Remove(block)) throw new InvalidOperationException("tree freed a block it did not hold"); }
        public bool HoldsModelPagedKv(KvBlock block) => _flags[block].Paged;
        public bool HoldsSnapshotBytes(KvBlock block) => _flags[block].Snapshot;
        public int UsedTokens(KvBlock block) => _flags[block].Snapshot ? BlockSize : 0;
        public int RefCount(KvBlock block) => _refs[block];
    }

    private sealed class WaitingView : IWaitingPlanView
    {
        public List<Request> Waiting = new();
        public Request? Current;

        public bool NoOtherWaiterTargets(int scopeIx, RadixNode node, long treeVersion)
        {
            int others = 0;
            foreach (Request w in Waiting)
            {
                if (ReferenceEquals(w, Current) || w.ScopeIx != scopeIx) continue;
                if (++others > 128) return false;
                if (w.Plan.Version != treeVersion || ReferenceEquals(w.Plan.PayloadNode, node)) return false;
            }
            return true;
        }
    }

    private readonly int _seed;
    private readonly Random _rng;
    private readonly Profile _profile;
    private readonly bool _tiny;
    private readonly bool _faults;
    private readonly PrefixTree _tree;
    private readonly RecordingHost _host = new();
    private readonly FakeValidator _validator = new();
    private readonly PrefixTreeInvariantChecker _checker = new();
    private readonly WaitingView _waiting = new();
    private readonly List<Session> _sessions = new();
    private readonly List<Request> _running = new();
    private readonly Dictionary<string, ResourceVector> _liveKeys = new(StringComparer.Ordinal);
    private readonly List<(ScopeId Scope, long[] Key, MediaSpanRecord[] Spans)> _computed = new();
    private readonly List<(long[] Prefix, MediaSpanRecord[] Spans, int Boundary)> _publicPrefixes = new();
    private readonly List<string> _log = new();
    private readonly int[][] _systemPrompts;
    private readonly string[] _mediaIds;
    private long _nowMs;
    private int _admittingReceipts;
    private string _op = "";
    internal int Plans, Comparisons, Acquires, Finishes, Evictions;

    internal TreeTraceHarness(int seed)
    {
        _seed = seed;
        _rng = new Random(seed);
        _profile = Profiles()[_rng.Next(Profiles().Length)];
        _tiny = _rng.Next(2) == 0;
        _faults = seed % 3 != 0;
        int[] s1 = Enumerable.Range(1000, 12).ToArray();
        _systemPrompts = new[] { s1, s1.Concat(Enumerable.Range(1100, 8)).ToArray(), Enumerable.Range(1200, 16).ToArray() };
        (string a, string b) = Tk.CollidingIds();
        _mediaIds = new[] { a, b, new string('c', 64), "path:/img/local.png" };
        long spare = _tiny ? 1500 : -1;
        _tree = new PrefixTree(new PrefixTreeOptions
        {
            Capabilities = _profile.Caps,
            BlockSize = BlockSize,
            PageHost = _host,
            PayloadValidator = _validator,
            PublicMax = _tiny ? 1 : 2,
            ScopedEndStateLeavesMax = _tiny ? 3 : 0,
            BatchedPagedEnabled = _profile.BatchedPaged,
            PageHostBytes = PageHostBytes,
            QuerySpareBytes = c => c == ResourceClass.PoolPages ? -1 : spare,
            PoolPagesCap = _tiny ? 12 : 0,
            ClockMs = () => _nowMs,
            TrackReceipts = true,
            StrictReceipts = true,
            ContextLength = 4096,
            EngineSerial = seed,
            TruncationSearchNodes = 64,
        });
        _validator.Rule = _profile.ModelRule;
    }

    private static Profile[] Profiles() => new[]
    {
        new Profile { Name = "OracleP", BatchedPaged = true, Caps = Caps(EndStateSupport.CopyAndDonate, TruncationKind.Any, 0, 1, PageSupport.Both, copy: true) },
        new Profile { Name = "OracleP2", BatchedPaged = true, Caps = Caps(EndStateSupport.None, TruncationKind.Any, 0, 1, PageSupport.Both) },
        new Profile { Name = "OracleS", Caps = Caps(EndStateSupport.CopyAndDonate, TruncationKind.WithinUnwrappedWindow, 64, 1, PageSupport.None, mmMin: 24) },
        new Profile { Name = "OracleS2", Caps = Caps(EndStateSupport.None, TruncationKind.WithinRingSlack, 4, 1, PageSupport.A1HostSlab, window: 32) },
        new Profile { Name = "OracleR", Caps = Caps(EndStateSupport.CopyAndDonate, TruncationKind.None, 0, 1, PageSupport.None, acrossMedia: false) },
        new Profile { Name = "OracleR2", Caps = Caps(EndStateSupport.None, TruncationKind.None, 0, 1, PageSupport.A1HostSlab, stateAtEnd: true) },
        new Profile
        {
            Name = "OracleN", NativeSlots = true,
            Caps = Caps(EndStateSupport.DonateOnly, TruncationKind.ModelDecides, 0, 2, PageSupport.None, nativeSlots: 3),
            ModelRule = (key, payload, target) => payload - target <= 8,
        },
    };

    private static PrefixCacheCapabilities Caps(EndStateSupport endState, TruncationKind truncation, int parameter, int granularity, PageSupport pages,
                                                bool copy = false, int mmMin = 0, int window = 0, bool acrossMedia = true, bool stateAtEnd = false, int nativeSlots = 0) => new()
    {
        Class = FamilyClass.P,
        Readiness = PrefixCacheMode.Legacy,
        NamespaceFingerprint = "harness",
        EndState = endState,
        CanCaptureCopy = endState == EndStateSupport.CopyAndDonate,
        AdoptPrimaryOnDisplacement = endState == EndStateSupport.CopyAndDonate,
        PrimaryResident = true,
        MinRetainTokens = MinRetain,
        Truncation = truncation,
        TruncationParameter = parameter,
        TruncationGranularity = granularity,
        RewindCapTokens = 16,
        Pages = pages,
        PagesNeedStateAtEnd = stateAtEnd,
        PageWindowTokens = window,
        SupportsCopyPagedToHolder = copy,
        ReuseAcrossMediaSpan = acrossMedia,
        MmReuseMinTokens = mmMin,
        MaxRetainedNativeSlots = nativeSlots,
    };

    internal string Describe() => $"seed {_seed} profile {_profile.Name} tiny={_tiny} faults={_faults}";

    // ------------------------------------------------------------------ driver

    internal void Run()
    {
        try
        {
            for (int step = 0; step < Steps; step++)
            {
                _nowMs += 1000 + _rng.Next(120_000);
                StepOnce();
                CheckAll(afterTrigger: false);
            }
            // Quiescence: finish everything, drain, and every lock is gone.
            _op = "quiesce";
            while (_running.Count > 0) Finish(_running[0], abort: _rng.Next(2) == 0);
            Drain();
            CheckAll(afterTrigger: false);
            Expect(_tree.Ledger!.Count == 0, "receipts remain open at quiescence");
            IReadOnlyList<InvariantViolation> q = _checker.Check(_tree, new InvariantCheckContext(Quiescent: true));
            Expect(q.Count == 0, "quiescence invariants: " + string.Join("; ", q));
        }
        catch (Exception ex) when (ex is not HarnessFailure)
        {
            throw new HarnessFailure($"{Describe()}: exception in op '{_op}': {ex}\ntrace tail:\n{Tail()}");
        }
    }

    private void StepOnce()
    {
        // Keep concurrency realistic (the engine runs a handful of requests at once).
        while (_running.Count > 8) Finish(_running[_rng.Next(_running.Count)], abort: _rng.Next(6) == 0);
        int roll = _rng.Next(157);
        if (roll >= 150)
        {
            if (roll < 153) { _op = "replay-flatten"; Admit(FlattenReplayRequest(), fault: MaybeFault()); }
            else { _op = "image-swap"; Admit(ImageSwapRequest(), fault: MaybeFault()); }
            _log.Add($"{_op} running={_running.Count} nodes={_tree.NodeCount} v={_tree.Version}");
            return;
        }
        if (roll < 10) { _op = "new-chat"; Admit(NewChatRequest(), fault: MaybeFault()); }
        else if (roll < 35) { _op = "follow-up"; Admit(FollowUpRequest(image: false, resend: false, breakpoint: false), fault: MaybeFault()); }
        else if (roll < 40) { _op = "regenerate"; Admit(RegenerateRequest(), fault: MaybeFault()); }
        else if (roll < 45) { _op = "fork-edit"; Admit(ForkRequest(), fault: MaybeFault()); }
        else if (roll < 50) { _op = "replay-new-scope"; Admit(ReplayRequest(lineage: false), fault: MaybeFault()); }
        else if (roll < 52) { _op = "replay-lineage"; Admit(ReplayRequest(lineage: true), fault: MaybeFault()); }
        else if (roll < 57) { _op = "image-new"; Admit(FollowUpRequest(image: true, resend: false, breakpoint: false), fault: MaybeFault()); }
        else if (roll < 60) { _op = "image-resend"; Admit(FollowUpRequest(image: true, resend: true, breakpoint: false), fault: MaybeFault()); }
        else if (roll < 63) { _op = "breakpoint"; Admit(FollowUpRequest(image: false, resend: false, breakpoint: true), fault: MaybeFault()); }
        else if (roll < 73) { _op = "burst"; Burst(identicalSystem: false); }
        else if (roll < 76) { _op = "burst-same-system"; Burst(identicalSystem: true); }
        else if (roll < 96) { _op = "finish"; if (_running.Count > 0) Finish(_running[_rng.Next(_running.Count)], abort: false); }
        else if (roll < 104) { _op = "abort"; if (_running.Count > 0) Finish(_running[_rng.Next(_running.Count)], abort: true); }
        else if (roll < 108) { _op = "preempt"; if (_running.Count > 0) Preempt(_running[_rng.Next(_running.Count)]); }
        else if (roll < 111) { _op = "pressure"; Pressure(); }
        else if (roll < 114) { _op = "retire"; Retire(); }
        else if (roll < 115) { _op = "backend-recreate"; _tree.InvalidateDeviceState(); }
        else if (roll < 116) { _op = "reset"; ResetTree(); }
        else if (roll < 124) { _op = "fault"; Fault(); }
        else if (roll < 132) { _op = "publish"; Publish(); }
        else if (roll < 140) { _op = "evict"; EvictRandom(); }
        else { _op = "drain"; Drain(); }
        _log.Add($"{_op} running={_running.Count} nodes={_tree.NodeCount} v={_tree.Version}");
    }

    private int MaybeFault() => _faults && _rng.Next(12) == 0 ? 1 + _rng.Next(3) : 0;

    // ------------------------------------------------------------------ request construction

    private int[] UserText(int min, int max)
    {
        int n = min + _rng.Next(max - min + 1);
        if (n > 0 && _rng.Next(8) == 0)
        {
            // A user pastes part of a system prompt: text that matches public content past someone's P.
            int[] sys = _systemPrompts[_rng.Next(_systemPrompts.Length)];
            int offset = _rng.Next(sys.Length);
            return sys.Skip(offset).Take(Math.Max(n, 3)).ToArray();
        }
        var t = new int[n];
        for (int i = 0; i < n; i++) t[i] = _rng.Next(20);    // a small vocabulary: accidental shared prefixes
        return t;
    }

    /// <summary>The same scope continues without a system prompt, carrying the old prompt as plain text (P = 0).</summary>
    private Request FlattenReplayRequest()
    {
        Session src = AnySession();
        if (src.Kind == ScopeKind.Unscoped) return ReplayRequest(lineage: true);
        Session s = NewSession(src.Kind);
        s.Id = src.Id;
        s.System = Array.Empty<int>();
        s.Transcript.AddRange(src.System);
        s.Transcript.AddRange(src.Transcript);
        s.Spans.AddRange(src.Spans);
        var transcript = new List<int>(s.Transcript);
        transcript.AddRange(UserText(1, 5));
        return Build(s, transcript, new List<MediaSpanRecord>(s.Spans));
    }

    /// <summary>The last image is replaced at the same position: another id (a 63-bit colliding one when possible), or one token longer.</summary>
    private Request ImageSwapRequest()
    {
        Session s = AnySession();
        if (s.Spans.Count == 0) return FollowUpRequest(image: true, resend: false, breakpoint: false);
        MediaSpanRecord last = s.Spans[^1];
        int keep = last.Start - s.System.Length;
        if (keep < 0 || keep > s.Transcript.Count) return FollowUpRequest(image: true, resend: false, breakpoint: false);
        var transcript = s.Transcript.Take(keep).ToList();
        var spans = s.Spans.Take(s.Spans.Count - 1).ToList();
        MediaId256 a = MediaId256.FromContentId(_mediaIds[0]), b = MediaId256.FromContentId(_mediaIds[1]);
        MediaId256 id = last.Id == a ? b : last.Id == b ? a : MediaId256.FromContentId(_mediaIds[_rng.Next(_mediaIds.Length)]);
        int len = last.End - last.Start + (_rng.Next(3) == 0 ? 1 : 0);
        if (_rng.Next(3) == 0) id = last.Id;   // same image, maybe a different length
        for (int i = 0; i < len; i++) transcript.Add(Placeholder);
        spans.Add(new MediaSpanRecord(last.Start, last.Start + len, id));
        transcript.AddRange(UserText(1, 6));
        return Build(s, transcript, spans);
    }

    private Session NewSession(ScopeKind? kind = null)
    {
        var s = new Session
        {
            Kind = kind ?? (ScopeKind)new[] { (int)ScopeKind.Session, (int)ScopeKind.Lineage, (int)ScopeKind.Fresh, (int)ScopeKind.Unscoped }[_rng.Next(4)],
            System = _rng.Next(4) switch { 0 => Array.Empty<int>(), var i => _systemPrompts[i - 1] },
        };
        s.Id = FreshId();
        if (_sessions.Count >= 8) _sessions.RemoveAt(_rng.Next(_sessions.Count));
        _sessions.Add(s);
        return s;
    }

    private ScopeId FreshId() => new(new UInt128((ulong)_rng.NextInt64(), (ulong)_rng.NextInt64() | 1));

    private Session AnySession() => _sessions.Count == 0 || _rng.Next(6) == 0 ? NewSession() : _sessions[_rng.Next(_sessions.Count)];

    private Request Build(Session s, List<int> transcript, List<MediaSpanRecord> spans, int breakpoint = 0)
    {
        var tokens = new List<int>(s.System);
        tokens.AddRange(transcript);
        var r = new Request
        {
            Session = s,
            P = s.System.Length,
            Tokens = tokens.ToArray(),
            Spans = spans.OrderBy(x => x.Start).ToArray(),
            Route = (ExpectedRoute)_rng.Next(_profile.BatchedPaged ? 3 : 2),
        };
        if (s.Kind == ScopeKind.Unscoped)
        {
            r.ScopeId = FreshId();
            r.ScopeIx = _tree.InternScope(r.ScopeId, ScopeKind.Unscoped);
        }
        else
        {
            r.ScopeId = s.Id;
            r.ScopeIx = _tree.InternScope(s.Id, s.Kind);
        }
        r.Key = RadixKeyBuilder.BuildPrompt(r.Tokens, r.Spans, _tree.KeyPool);
        r.Limit = r.Tokens.Length - 1;
        if (breakpoint > 0 && breakpoint < r.Limit)
        {
            r.Limit = breakpoint;
            r.BreakpointLimit = breakpoint;
        }
        return r;
    }

    private Request NewChatRequest(Session? s = null)
    {
        s ??= NewSession();
        s.Transcript.Clear();
        s.Spans.Clear();
        var transcript = new List<int>(UserText(3, 14));
        return Build(s, transcript, new List<MediaSpanRecord>());
    }

    private Request FollowUpRequest(bool image, bool resend, bool breakpoint)
    {
        Session s = AnySession();
        if (s.System.Length + s.Transcript.Count > 220) return NewChatRequest(s);
        var transcript = new List<int>(s.Transcript);
        var spans = new List<MediaSpanRecord>(s.Spans);
        transcript.AddRange(UserText(1, 6));
        if (image)
        {
            string id = resend && s.Spans.Count > 0 ? _mediaIds[_rng.Next(_mediaIds.Length)] : _mediaIds[_rng.Next(_mediaIds.Length)];
            if (!resend && _rng.Next(2) == 0) id = $"{_rng.NextInt64():x16}{_rng.NextInt64():x16}{_rng.NextInt64():x16}{_rng.NextInt64():x16}";
            int start = s.System.Length + transcript.Count;
            int len = 3 + _rng.Next(10);
            for (int i = 0; i < len; i++) transcript.Add(Placeholder);
            spans.Add(MediaSpanRecord.FromContentId(start, start + len, id));
        }
        transcript.AddRange(UserText(1, 8));
        int bp = breakpoint ? s.System.Length + _rng.Next(Math.Max(1, transcript.Count)) : 0;
        return Build(s, transcript, spans, bp);
    }

    private Request RegenerateRequest()
    {
        Session s = AnySession();
        int drop = Math.Min(s.Transcript.Count, 1 + _rng.Next(12));
        var transcript = s.Transcript.Take(s.Transcript.Count - drop).ToList();
        var spans = s.Spans.Where(x => x.End <= s.System.Length + transcript.Count).ToList();
        transcript.AddRange(UserText(1, 4));
        return Build(s, transcript, spans);
    }

    private Request ForkRequest()
    {
        Session s = AnySession();
        int keep = s.Transcript.Count == 0 ? 0 : _rng.Next(s.Transcript.Count);
        var transcript = s.Transcript.Take(keep).ToList();
        var spans = s.Spans.Where(x => x.End <= s.System.Length + keep).ToList();
        transcript.AddRange(UserText(2, 10));
        return Build(s, transcript, spans);
    }

    private Request ReplayRequest(bool lineage)
    {
        Session src = AnySession();
        Session s = NewSession(lineage ? src.Kind == ScopeKind.Unscoped ? ScopeKind.Lineage : src.Kind : ScopeKind.Fresh);
        s.System = src.System;
        if (lineage && src.Kind != ScopeKind.Unscoped) s.Id = src.Id;        // possession-proven lineage: same scope
        s.Transcript.AddRange(src.Transcript);
        s.Spans.AddRange(src.Spans);
        var transcript = new List<int>(s.Transcript);
        transcript.AddRange(UserText(1, 5));
        return Build(s, transcript, new List<MediaSpanRecord>(s.Spans));
    }

    // ------------------------------------------------------------------ admission

    private void Admit(Request r, int fault)
    {
        _tree.NoteRequests(r.ScopeIx, +1, 0);
        _waiting.Waiting.Add(r);
        PlanChecked(r);
        AdmitPlanned(r, fault);
    }

    private void Burst(bool identicalSystem)
    {
        int n = 2 + _rng.Next(5);
        var batch = new List<Request>();
        int[]? system = identicalSystem ? _systemPrompts[_rng.Next(_systemPrompts.Length)] : null;
        for (int i = 0; i < n; i++)
        {
            Request r;
            if (identicalSystem)
            {
                Session s = NewSession();
                s.System = system!;
                r = NewChatRequest(s);
            }
            else
            {
                r = _rng.Next(3) switch { 0 => ForkRequest(), 1 => FollowUpRequest(false, false, false), _ => NewChatRequest() };
            }
            _tree.NoteRequests(r.ScopeIx, +1, 0);
            _waiting.Waiting.Add(r);
            batch.Add(r);
        }
        foreach (Request r in batch) PlanChecked(r);
        // FCFS admission: a plan made at an older version is re-planned first (P2).
        foreach (Request r in batch)
        {
            if (r.Plan.Version != _tree.Version) PlanChecked(r);
            AdmitPlanned(r, MaybeFault());
            CheckAll(afterTrigger: false);
        }
    }

    private void PlanChecked(Request r)
    {
        _waiting.Current = r;
        var mr = new MatchRequest(r.Key, r.Tokens.Length, r.ScopeIx, r.P, r.Limit, r.Spans, r.Route, _running.Count == 0);
        long version = _tree.Version;
        int nodes = _tree.NodeCount;
        _tree.Plan(mr, r.Plan, _waiting);
        Plans++;
        // Purity.
        var again = new MatchPlan();
        _tree.Plan(mr, again, _waiting);
        Expect(_tree.Version == version && _tree.NodeCount == nodes, "Match/Evaluate mutated the tree");
        Expect(again.Kind == r.Plan.Kind && again.Mode == r.Plan.Mode && again.Length == r.Plan.Length
               && ReferenceEquals(again.PayloadNode, r.Plan.PayloadNode), $"Plan is not deterministic: {r.Plan} vs {again}");
        // Brute force.
        if (!r.Plan.SearchCapped && !r.Plan.TruncationSearchCapped)
        {
            Reference expected = ReferenceEvaluate(_tree, mr, _waiting, _validator);
            Comparisons++;
            string diff = expected.Diff(r.Plan);
            Expect(diff.Length == 0, $"plan differs from brute force: {diff}\n plan {r.Plan}\n request scope={r.ScopeIx} P={r.P} len={r.Tokens.Length} λ={r.Limit} spans={r.Spans.Length} route={r.Route} primaryAvailable={mr.PrimaryAvailable}");
        }
        CheckPlanProperties(r, mr);
        _waiting.Current = null;
    }

    private void CheckPlanProperties(Request r, in MatchRequest mr)
    {
        MatchPlan plan = r.Plan;
        long[] key = KeyArray(r.Key, r.Tokens.Length);
        // Isolation: reuse never exceeds what this scope, or a public prefix within the public cap, computed.
        int bound = AllowedBound(r, key);
        Expect(plan.Length <= bound, $"isolation: reused {plan.Length} > allowed bound {bound} ({plan})");
        Expect(plan.Length <= Math.Max(0, mr.KeyLength - 1), "K6: at least one token must stay un-matched");
        Expect(plan.Length <= r.Limit, "reuse exceeds the match limit (K7)");
        // Media-span atomicity (M4) and the M6 clamp.
        foreach (MediaSpanRecord s in r.Spans)
            Expect(!(s.Start < plan.Length && plan.Length < s.End), $"reuse {plan.Length} ends inside media span [{s.Start},{s.End})");
        if (!_profile.Caps.ReuseAcrossMediaSpan && r.Spans.Length > 0)
            Expect(plan.Length <= r.Spans[0].Start, "reuse crosses a media span on a family that cannot continue past one");
        if (!plan.HasReuse) return;
        // Scope: every node on the path to L is public or own; public-only reuse stays within the cap.
        RadixNode anchor = plan.AnchorParent!;
        Expect(anchor.EdgeStartDepth < plan.Length && plan.Length <= anchor.Depth, "the anchor does not contain the plan length");
        bool anyOwn = false;
        for (RadixNode? n = anchor; n is not null && !n.IsRoot; n = n.Parent)
        {
            Expect(n.ScopeIx == 0 || n.ScopeIx == r.ScopeIx, $"plan path crosses scope {n.ScopeIx} (request scope {r.ScopeIx})");
            anyOwn |= n.ScopeIx != 0;
        }
        if (plan.PayloadNode is { } x)
        {
            Expect(x.ScopeIx == 0 || x.ScopeIx == r.ScopeIx, "payload of another scope");
            if (x.ScopeIx == 0) Expect(plan.Length <= plan.PublicCap, "public payload used past the public cap");
        }
        if (plan.Kind == CandidateKind.Pages && anchor.ScopeIx == 0 && !anyOwn)
            Expect(plan.Length <= plan.PublicCap, "public pages used past the public cap");
        Expect(plan.PublicTokens <= plan.Length && plan.PublicTokens <= plan.PublicCap, "PublicTokens out of range");
    }

    private void AdmitPlanned(Request r, int fault)
    {
        _waiting.Current = r;
        MatchPlan plan = r.Plan;
        LockReceipt receipt = _tree.Acquire(plan);
        Acquires++;
        if (plan.HasReuse)
        {
            RadixNode anchor = receipt.PathAnchor!;
            Expect(anchor.Depth == plan.Length, $"acquired anchor depth {anchor.Depth} != plan length {plan.Length}");
            long[] path = PrefixTree.PathKey(anchor);
            for (int i = 0; i < path.Length; i++)
                Expect(path[i] == r.Key[i], $"acquired path differs from the key at {i}");
            if (plan.PayloadNode is not null) Expect(ReferenceEquals(receipt.StateAnchor, plan.PayloadNode), "state lock is not on the payload");
        }
        _admittingReceipts = receipt.IsEmpty ? 0 : 1;
        CheckAll(afterTrigger: false, transactionOpen: false);
        if (fault != 0 && plan.HasReuse)
        {
            // Clone failure, allocation failure after lock, or truncation decline: roll back (P3).
            _tree.Release(ref receipt);
            _admittingReceipts = 0;
            _waiting.Waiting.Remove(r);
            _tree.NoteRequests(r.ScopeIx, -1, 0);
            _tree.ReleaseRopeOwner(r.Key);
            _tree.FlushQueuedInvalidations();
            _log.Add($"  rollback fault={fault}");
            return;
        }
        RadixNode? x = plan.PayloadNode;
        switch (plan.Mode)
        {
            case MaterializeMode.DonateEndState:
            case MaterializeMode.KeepPrimary:
                _tree.MarkDonationPending(x!);
                CheckAll(afterTrigger: false, transactionOpen: true);
                string key = x!.EndState!.Key;
                _tree.CommitDonation(x);
                Expect(_liveKeys.Remove(key), $"donated key {key} was not live");
                _tree.ReleaseState(ref receipt);
                break;
            case MaterializeMode.CloneEndState:
            case MaterializeMode.ConvertPrimaryThenClone:
                _tree.ReleaseState(ref receipt);
                break;
        }
        _admittingReceipts = 0;
        r.Durable = receipt;
        r.Reused = plan.Length;
        _waiting.Waiting.Remove(r);
        _tree.NoteRequests(r.ScopeIx, -1, +1);
        _running.Add(r);
        _tree.FlushQueuedInvalidations();
        _waiting.Current = null;
    }

    // ------------------------------------------------------------------ finish, preemption, publication

    private void Finish(Request r, bool abort, NodeFlags extra = NodeFlags.None)
    {
        _running.Remove(r);
        Finishes++;
        if (!abort)
        {
            int[] output = UserText(0, 10);
            RadixKeyBuilder.AppendOutput(r.Key, r.Tokens.Length, output, _tree.KeyPool);
            int total = r.Tokens.Length + output.Length;
            int computed = _rng.Next(4) == 0 ? Math.Max(1, r.Reused + _rng.Next(Math.Max(1, total - r.Reused))) : total;
            computed = Math.Clamp(computed, 1, total);
            if (r.BreakpointLimit > 0) computed = Math.Min(computed, r.BreakpointLimit);
            InsertComputed(r, computed, extra);
            if (!_tiny || _rng.Next(2) == 0)
            {
                r.Session.Transcript.Clear();
                r.Session.Transcript.AddRange(r.Tokens.Skip(r.P));
                r.Session.Transcript.AddRange(output);
                r.Session.Spans.Clear();
                r.Session.Spans.AddRange(r.Spans);
            }
        }
        _tree.Release(ref r.Durable);
        _tree.NoteRequests(r.ScopeIx, 0, -1);
        _tree.ReleaseRopeOwner(r.Key);
        Triggers();
    }

    private void Preempt(Request r) => Finish(r, abort: false, extra: NodeFlags.PreemptHold);

    private bool ScopeUsable(Request r) => _tree.Scopes.IsLive(r.ScopeIx) && !_tree.Scopes[r.ScopeIx].Retired
                                           && _tree.Scopes[r.ScopeIx].Id == r.ScopeId;

    private void InsertComputed(Request r, int computed, NodeFlags flags)
    {
        if (computed < MinRetain || !ScopeUsable(r)) return;
        PrefixCacheCapabilities caps = _profile.Caps;
        bool inserted = false;
        // Pages: whole pages only.
        if (caps.Pages != PageSupport.None && computed >= BlockSize)
        {
            int pageLen = computed / BlockSize * BlockSize;
            RadixNode n = _tree.Insert(r.Key, pageLen, r.ScopeIx, r.P, NodeFlags.None, r.Spans);
            if (n.Depth == pageLen)
            {
                var pages = new PageRef[pageLen / BlockSize];
                for (int p = 0; p < pages.Length; p++)
                {
                    int kind = _rng.Next(10);
                    PageStore store = caps.Pages == PageSupport.A1HostSlab ? PageStore.A1HostSlab
                        : kind < 4 ? PageStore.Both : kind < 7 ? PageStore.A1HostSlab : PageStore.A2ModelPaged;
                    bool unbacked = _faults && _rng.Next(25) == 0;
                    KvBlock block = unbacked ? _host.New(false, false) : _host.New((store & PageStore.A2ModelPaged) != 0, (store & PageStore.A1HostSlab) != 0);
                    pages[p] = new PageRef(block, p, store, StateAtEnd: !caps.PagesNeedStateAtEnd || _rng.Next(3) == 0 || p == pages.Length - 1);
                }
                _tree.AttachPages(n, pages);
                foreach (PageRef p in pages) _host.DropCaller(p.Block);
                inserted = true;
            }
            _tree.CollectIfEmpty(n);
        }
        // End state: the primary resident or a donated holder / native slot.
        bool primary = caps.PrimaryResident && _tree.PrimaryResidentCount == 0 && _rng.Next(4) == 0;
        if (caps.EndState != EndStateSupport.None || primary)
        {
            RadixNode n = _tree.Insert(r.Key, computed, r.ScopeIx, r.P, flags, r.Spans);
            if (n.Depth == computed)
            {
                inserted = true;
                if (!(_faults && _rng.Next(15) == 0))   // capture failure: nothing attached
                {
                    EndStatePayload payload = primary && n.EndState is null
                        ? new EndStatePayload { Key = _tree.MintKey(), Kind = EndStateKind.PrimaryResident }
                        : Holder(computed);
                    if (payload.Kind == EndStateKind.PrimaryResident || caps.EndState != EndStateSupport.None)
                    {
                        AttachResult result = _tree.AttachEndState(n, payload);
                        if (result != AttachResult.Refused) _liveKeys[payload.Key] = payload.Bytes;
                        if (result is AttachResult.Attached or AttachResult.Revived) LivenessProbe(r, computed, payload);
                    }
                }
            }
            _tree.CollectIfEmpty(n);
        }
        if (inserted)
        {
            long[] key = KeyArray(r.Key, computed);
            _computed.Add((r.ScopeId, key, r.Spans.Where(s => s.Start < computed).ToArray()));
            if (r.P > 0)
            {
                int pub = Math.Min(r.P, computed);
                _publicPrefixes.Add((key.Take(pub).ToArray(), r.Spans.Where(s => s.Start < pub).ToArray(), r.P <= computed ? r.P : 0));
            }
        }
    }

    private EndStatePayload Holder(int tokens)
    {
        long scale = _tiny ? 60 : 10;
        var bytes = _profile.NativeSlots
            ? new ResourceVector { NativeSlot = 200 * scale }
            : new ResourceVector { HostKv = tokens * scale, DeviceKv = _rng.Next(2) * tokens * scale, StateSnapshot = _rng.Next(2) * 50 * scale };
        return new EndStatePayload
        {
            Key = _tree.MintKey(),
            Kind = _profile.NativeSlots ? EndStateKind.NativeSlot : EndStateKind.Holder,
            Origin = PayloadOrigin.Donation,
            Footprint = new PayloadFootprint(tokens, tokens, bytes, 0),
            DeviceDirty = bytes.DeviceKv > 0,
        };
    }

    /// <summary>Liveness (D2/D3 guard): in fault-free, unbounded runs, a same-scope follow-up reuses the end state just stored.</summary>
    private void LivenessProbe(Request r, int computed, EndStatePayload payload)
    {
        if (_faults || _tiny || _profile.Caps.EndState == EndStateSupport.DonateOnly || _profile.Caps.MmReuseMinTokens > 0) return;
        // A primary resident that cannot be converted is only reusable by donation (a leaf, unlocked, scoped).
        if (payload.Kind == EndStateKind.PrimaryResident
            && !(_profile.Caps.AdoptPrimaryOnDisplacement && _profile.Caps.EndState == EndStateSupport.CopyAndDonate)) return;
        var tokens = KeyArray(r.Key, computed).ToList();
        var probeKey = new KeyRope();
        probeKey.Append(tokens.ToArray(), _tree.KeyPool);
        probeKey.Append(7, _tree.KeyPool);
        var mr = new MatchRequest(probeKey, computed + 1, r.ScopeIx, r.P, computed, r.Spans.Where(s => s.End <= computed).ToArray(),
                                  ExpectedRoute.PerSequenceFused, PrimaryAvailable: true);
        if (_tree.Rules.ClampLength(computed, mr) != computed || r.Spans.Any(s => s.Start < computed && computed < s.End)) { probeKey.ReturnChunks(_tree.KeyPool); return; }
        var plan = new MatchPlan();
        _tree.Plan(mr, plan);
        Expect(plan.Length >= computed, $"liveness: a same-scope follow-up reused {plan.Length} < stored {computed} ({plan})");
        probeKey.ReturnChunks(_tree.KeyPool);
    }

    private void Publish()
    {
        Request? r = _running.Where(x => !x.Published && x.P >= MinRetain).OrderBy(_ => _rng.Next()).FirstOrDefault();
        if (r is null || !_profile.Caps.CanCaptureCopy) return;
        r.Published = true;
        if (!ScopeUsable(r)) return;
        RadixNode n = _tree.Insert(r.Key, r.P, 0, r.P, NodeFlags.None, r.Spans);
        if (n.Depth == r.P)
        {
            EndStatePayload payload = Holder(r.P);
            AttachResult result = _tree.AttachEndState(n, payload);
            if (result != AttachResult.Refused) _liveKeys[payload.Key] = payload.Bytes;
            long[] key = KeyArray(r.Key, r.P);
            _publicPrefixes.Add((key, Array.Empty<MediaSpanRecord>(), r.P));
            _computed.Add((r.ScopeId, key, Array.Empty<MediaSpanRecord>()));
            // Page publication moves the durable lock to the deepest published node (P5).
            if (_rng.Next(2) == 0 && r.Durable.PathDepth <= r.P) _tree.MoveDurableLock(ref r.Durable, n);
        }
        _tree.CollectIfEmpty(n);
        Triggers();
    }

    // ------------------------------------------------------------------ pressure, retirement, faults

    private void Triggers()
    {
        if (!_tiny) return;
        bool ok = _tree.EnforceCountSubCaps();
        ok &= _tree.EnforceCaps(EvictionTier.ScopeNewest);
        Evictions++;
        CheckAll(afterTrigger: ok);
    }

    private void Pressure()
    {
        _tree.RelieveMemoryPressure(_rng.Next(2) == 0 ? PressureLevel.Moderate : PressureLevel.Critical);
    }

    private void EvictRandom()
    {
        var cls = (ResourceClass)_rng.Next(ResourceVector.ClassCount);
        _tree.Evict(cls, 1 + _rng.Next(_tiny ? 400 : 40), ReleaseReason.Evicted, (EvictionTier)_rng.Next(EvictionLists.TierCount));
        Evictions++;
    }

    private void Retire()
    {
        if (_sessions.Count == 0) return;
        Session s = _sessions[_rng.Next(_sessions.Count)];
        _tree.RetireScope(s.Id);
        s.Id = FreshId();          // new chat epoch
        s.Transcript.Clear();
        s.Spans.Clear();
    }

    private void Fault()
    {
        List<string> keys = _tree.PayloadKeys.ToList();
        if (keys.Count == 0 || !_faults) return;
        string key = keys[_rng.Next(keys.Count)];
        if (_rng.Next(2) == 0)
        {
            // The model reclaimed a payload on its own (reported through the sink, drained later).
            if (_tree.TryGetNodeByKey(key, out RadixNode n) && n.StateLockRef == 0 && !n.IsDonationPending)
            {
                _tree.InvalidatePayload(key);
                _liveKeys.Remove(key);
            }
        }
        else
        {
            _validator.Refused.Add(key);   // CanMaterialize flips to false
        }
    }

    private void ResetTree()
    {
        _tree.Reset();
        // Every running request's receipt is stale now: the engine drops them (it rebuilt its state).
        foreach (Request r in _running) r.Durable = default;
    }

    private void Drain()
    {
        int calls = 0;
        _tree.DrainReclaimQueue((keys, reason) =>
        {
            calls++;
            foreach (string k in keys)
                Expect(_liveKeys.Remove(k), $"released key {k} was not live");
        });
        Expect(calls <= 1, "a drain made more than one release call");
    }

    // ------------------------------------------------------------------ checks

    private void CheckAll(bool afterTrigger, bool transactionOpen = false)
    {
        IReadOnlyList<InvariantViolation> v = _checker.Check(_tree, new InvariantCheckContext(TransactionOpen: transactionOpen, AfterEvictionTrigger: afterTrigger));
        Expect(v.Count == 0, "invariants: " + string.Join("; ", v.Take(5)));
        // Receipt balance: the ledger holds exactly the receipts the requests hold.
        int held = _running.Count(r => !r.Durable.IsEmpty) + _admittingReceipts;
        Expect(_tree.Ledger!.Count == held, $"receipt balance: ledger {_tree.Ledger.Count} != held {held}");
        // Byte conservation: model-held payload bytes == tree end-state bytes + pending reclaim.
        ResourceVector keyBytes = default;
        foreach (ResourceVector b in _liveKeys.Values) keyBytes += b;
        var pageBytes = new ResourceVector { PoolPages = _host.TreeHeld.Count, HostKv = _host.TreeHeldSnapshots * PageHostBytes };
        Expect(keyBytes == _tree.Cached - pageBytes + _tree.PendingReclaim,
               $"byte conservation: live payload keys {keyBytes} != cached {_tree.Cached} - pages {pageBytes} + pending {_tree.PendingReclaim}");
        int treeKeys = _tree.PayloadKeys.Count + _tree.Reclaim.Count;
        Expect(_liveKeys.Count == treeKeys, $"key-set size: live {_liveKeys.Count} != tree {_tree.PayloadKeys.Count} + queued {_tree.Reclaim.Count}");
        foreach (string k in _tree.PayloadKeys) Expect(_liveKeys.ContainsKey(k), $"tree key {k} is not live");
        foreach (string k in _tree.Reclaim.Keys) Expect(_liveKeys.ContainsKey(k), $"queued key {k} is not live");
    }

    private int AllowedBound(Request r, long[] key)
    {
        int own = 0;
        foreach ((ScopeId scope, long[] seq, MediaSpanRecord[] spans) in _computed)
            if (scope == r.ScopeId) own = Math.Max(own, MediaLcp(key, r.Spans, seq, spans));
        int publicLcp = 0, boundary = 0;
        foreach ((long[] prefix, MediaSpanRecord[] spans, int b) in _publicPrefixes)
        {
            int lcp = MediaLcp(key, r.Spans, prefix, spans);
            publicLcp = Math.Max(publicLcp, lcp);
            if (b > 0 && lcp >= b) boundary = Math.Max(boundary, b);
        }
        return Math.Max(own, Math.Min(publicLcp, Math.Max(r.P, boundary)));
    }

    /// <summary>Longest common prefix of two keys where every span starting inside it must be identical on both sides.</summary>
    private static int MediaLcp(long[] a, MediaSpanRecord[] aSpans, long[] b, MediaSpanRecord[] bSpans)
    {
        int n = Math.Min(a.Length, b.Length);
        int m = 0;
        while (m < n && a[m] == b[m]) m++;
        foreach (MediaSpanRecord s in aSpans)
            if (s.Start < m && !bSpans.Contains(s)) m = Math.Min(m, s.Start);
        foreach (MediaSpanRecord s in bSpans)
            if (s.Start < m && !aSpans.Contains(s)) m = Math.Min(m, s.Start);
        return m;
    }

    private static long[] KeyArray(KeyRope rope, int length)
    {
        var a = new long[Math.Min(length, rope.Length)];
        for (int i = 0; i < a.Length; i++) a[i] = rope[i];
        return a;
    }

    private void Expect(bool condition, string message)
    {
        if (!condition)
            throw new HarnessFailure($"{Describe()}: {message}\nop '{_op}', trace tail:\n{Tail()}");
    }

    private string Tail() => string.Join("\n", _log.Skip(Math.Max(0, _log.Count - 25)));

    internal sealed class HarnessFailure : Exception
    {
        public HarnessFailure(string message) : base(message) { }
    }

    // ------------------------------------------------------------------ brute-force reference

    internal sealed class Reference
    {
        public CandidateKind Kind; public MaterializeMode Mode; public int Length; public RadixNode? Payload;
        public int PageCount; public int Structural; public int PublicCap;

        public string Diff(MatchPlan p)
        {
            var sb = new StringBuilder();
            if (p.Kind != Kind) sb.Append($"kind {p.Kind}≠{Kind} ");
            if (p.Mode != Mode) sb.Append($"mode {p.Mode}≠{Mode} ");
            if (p.Length != Length) sb.Append($"length {p.Length}≠{Length} ");
            if (p.PageCount != PageCount) sb.Append($"pages {p.PageCount}≠{PageCount} ");
            if (p.Structural != Structural) sb.Append($"structural {p.Structural}≠{Structural} ");
            if (p.PublicCap != PublicCap) sb.Append($"publicCap {p.PublicCap}≠{PublicCap} ");
            if (Kind is CandidateKind.EndState or CandidateKind.PrimaryResident or CandidateKind.TruncatedEndState && !ReferenceEquals(p.PayloadNode, Payload))
                sb.Append($"payload {p.PayloadNode}≠{Payload} ");
            return sb.ToString();
        }
    }

    private struct Cand
    {
        public CandidateKind Kind; public MaterializeMode Mode; public int Length; public RadixNode? Payload; public long Tie;
        public readonly bool Valid => Kind != CandidateKind.None && Length > 0;
    }

    /// <summary>
    /// Evaluates a request by brute force over every node: the matched length of each node is computed
    /// independently (element by element, spans checked at every start), then the §5.3 candidate rules
    /// are applied to every node — no trails, no DFS, no early exits.
    /// </summary>
    internal static Reference ReferenceEvaluate(PrefixTree t, in MatchRequest r, IWaitingPlanView? waiting, IPayloadValidator? validator)
    {
        PrefixCacheCapabilities caps = t.Caps;
        ResumabilityRules rules = t.Rules;
        int limit = ResumabilityRules.EffectiveMatchLimit(r);
        var lcp = new Dictionary<RadixNode, int>(ReferenceEqualityComparer.Instance);
        var order = new List<RadixNode>();
        var stack = new Stack<RadixNode>();
        stack.Push(t.Root);
        lcp[t.Root] = 0;
        while (stack.Count > 0)
        {
            RadixNode parent = stack.Pop();
            if (lcp[parent] != parent.Depth || parent.Depth >= limit) continue;   // children unreachable
            foreach (RadixNode child in parent.Children)
            {
                if (child.ScopeIx != 0 && child.ScopeIx != r.ScopeIx) continue;
                int start = child.EdgeStartDepth;
                int m = start;
                int max = Math.Min(child.Depth, Math.Min(limit, r.Key.Length));
                while (m < max && child.Edge[m - start] == r.Key[m]) m++;
                // spans starting inside [start, m): node records vs request spans
                MediaSpanRecord[] records = child.SpanRecords.ToArray();
                foreach (MediaSpanRecord rec in records)
                    if (rec.Start >= start && rec.Start < m && !(r.Spans ?? Array.Empty<MediaSpanRecord>()).Contains(rec)) m = Math.Min(m, rec.Start);
                foreach (MediaSpanRecord s in r.Spans ?? Array.Empty<MediaSpanRecord>())
                    if (s.Start >= start && s.Start < m && !records.Contains(s)) m = Math.Min(m, s.Start);
                if (m <= start) continue;
                lcp[child] = m;
                order.Add(child);
                stack.Push(child);
            }
        }
        var result = new Reference();
        int structural = 0, boundary = 0;
        foreach (RadixNode n in order)
        {
            structural = Math.Max(structural, lcp[n]);
            if (n.ScopeIx == 0 && n.IsPublicBoundary && lcp[n] == n.Depth) boundary = Math.Max(boundary, n.Depth);
        }
        int cap = Math.Max(r.PublicBoundary, boundary);
        result.Structural = structural;
        result.PublicCap = cap;

        // (A)
        Cand a = default;
        foreach (RadixNode n in order)
        {
            if (lcp[n] != n.Depth || n.EndState is null || n.IsDonationPending) continue;
            int len = n.Depth;
            bool primary = n.EndState.Kind == EndStateKind.PrimaryResident;
            if (!(n.ScopeIx == r.ScopeIx || (n.ScopeIx == 0 && len <= cap))) continue;
            if (rules.ClampLength(len, r) != len) continue;
            if (primary && (!r.PrimaryAvailable || n.StateLockRef > 0)) continue;
            if (!primary && validator is not null && !validator.CanMaterialize(n.EndState.Key, len, len)) continue;
            MaterializeMode mode = RefDonation(t, n, len, primary, waiting);
            if (mode == MaterializeMode.None) continue;
            a = Pick(a, new Cand { Kind = primary ? CandidateKind.PrimaryResident : CandidateKind.EndState, Mode = mode, Length = len, Payload = n, Tie = n.Id });
        }
        // (B)
        Cand b = default;
        if (caps.Pages != PageSupport.None || caps.SupportsCopyPagedToHolder)
        {
            foreach (RadixNode end in order)
            {
                var path = new List<RadixNode>();
                for (RadixNode? n = end; n is not null && !n.IsRoot; n = n.Parent) path.Add(n);
                path.Reverse();
                int structuralT = lcp[end];
                foreach (MaterializeMode mode in new[] { MaterializeMode.InjectA1Pages, MaterializeMode.BindPagesInPlace, MaterializeMode.CopyA2PagesToHolder })
                {
                    int usable = 0;
                    int p = 0;
                    bool stop = false;
                    foreach (RadixNode n in path)
                    {
                        while (!stop && (p + 1) * BlockSize - 1 < n.Depth)
                        {
                            int pageEnd = (p + 1) * BlockSize;
                            PageRef? page = null;
                            foreach (PageRef pr in n.PageSpan) if (pr.PageIndex == p) page = pr;
                            if (page is null) { stop = true; break; }
                            if (pageEnd > structuralT || !(n.ScopeIx == r.ScopeIx || (n.ScopeIx == 0 && pageEnd <= cap))
                                || !rules.RouteCanRead(page.Value.Store, mode, r.Route)) { stop = true; break; }
                            if (caps.PageWindowTokens > 0 && pageEnd > caps.PageWindowTokens) { stop = true; break; }
                            if ((!caps.PagesNeedStateAtEnd || page.Value.StateAtEnd) && rules.ClampLength(pageEnd, r) == pageEnd) usable = pageEnd;
                            p++;
                        }
                        if (stop) break;
                    }
                    if (usable <= 0) continue;
                    long pref = ResumabilityRules.PageModePreference(mode, r.Route);
                    if (!b.Valid || usable > b.Length || (usable == b.Length && pref < b.Tie))
                        b = new Cand { Kind = CandidateKind.Pages, Mode = mode, Length = usable, Tie = pref };
                }
            }
            if (b.Valid) b.Tie = long.MaxValue;
        }
        // (C)
        Cand c = default;
        if (caps.Truncation != TruncationKind.None)
        {
            int best = Math.Max(a.Valid ? a.Length : 0, b.Valid ? b.Length : 0);
            foreach (RadixNode stop in order)
            {
                bool isEnd = lcp[stop] < stop.Depth || lcp[stop] >= limit;
                if (!isEnd)
                {
                    isEnd = true;
                    foreach (RadixNode child in stop.Children)
                        if (lcp.ContainsKey(child)) { isEnd = false; break; }
                }
                if (!isEnd) continue;
                // Align and clamp until stable (a single AlignDown(Clamp(x)) can land inside a span).
                int target = rules.ClampLength(lcp[stop], r);
                while (true)
                {
                    int aligned = target - target % caps.TruncationGranularity;
                    int again = rules.ClampLength(aligned, r);
                    if (again == aligned) { target = aligned; break; }
                    target = again;
                }
                if (target <= best || target <= 0) continue;
                var queue = new Queue<RadixNode>();
                queue.Enqueue(stop);
                while (queue.Count > 0)
                {
                    RadixNode d = queue.Dequeue();
                    foreach (RadixNode child in d.Children)
                        if (child.ScopeIx == 0 || child.ScopeIx == r.ScopeIx) queue.Enqueue(child);
                    if (d.EndState is null || d.IsDonationPending || d.Depth <= target) continue;
                    if (!(d.ScopeIx == r.ScopeIx || (d.ScopeIx == 0 && target <= cap))) continue;
                    if (!rules.RewindWithinCap(d.Depth, target) || !rules.TruncationAllows(d.Depth, target)) continue;
                    bool primary = d.EndState.Kind == EndStateKind.PrimaryResident;
                    if (primary && (!r.PrimaryAvailable || d.StateLockRef > 0)) continue;
                    if (!primary && validator is not null && !validator.CanMaterialize(d.EndState.Key, d.Depth, target)) continue;
                    MaterializeMode mode = RefDonation(t, d, target, primary, waiting);
                    if (mode == MaterializeMode.None) continue;
                    var cand = new Cand { Kind = CandidateKind.TruncatedEndState, Mode = mode, Length = target, Payload = d, Tie = d.Id };
                    if (!c.Valid || cand.Length > c.Length
                        || (cand.Length == c.Length && (d.Depth < c.Payload!.Depth
                            || (d.Depth == c.Payload.Depth && (Rank(cand) < Rank(c) || (Rank(cand) == Rank(c) && cand.Tie < c.Tie))))))
                        c = cand;
                }
            }
        }
        Cand chosen = Pick(Pick(a, b), c);
        if (chosen.Valid && ResumabilityRules.IsClone(chosen.Mode) && chosen.Length < t.Options.MinCloneTokens)
        {
            Cand next = default;
            foreach (Cand x in new[] { a, b, c })
                if (x.Valid && !ResumabilityRules.IsClone(x.Mode) && !(x.Kind == chosen.Kind && x.Length == chosen.Length && ReferenceEquals(x.Payload, chosen.Payload)))
                    next = Pick(next, x);
            chosen = next;
        }
        if (chosen.Valid && caps.MmReuseMinTokens > 0 && chosen.Length < caps.MmReuseMinTokens && (r.Spans ?? Array.Empty<MediaSpanRecord>()).Any(s => s.Start >= chosen.Length))
            chosen = default;
        if (chosen.Valid)
        {
            result.Kind = chosen.Kind; result.Mode = chosen.Mode; result.Length = chosen.Length; result.Payload = chosen.Payload;
            result.PageCount = chosen.Kind == CandidateKind.Pages ? chosen.Length / BlockSize : 0;
        }
        return result;
    }

    private static int Rank(in Cand c) => ResumabilityRules.TieRank(c.Kind, c.Mode);

    private static Cand Pick(in Cand x, in Cand y)
    {
        if (!y.Valid) return x;
        if (!x.Valid) return y;
        if (x.Length != y.Length) return x.Length > y.Length ? x : y;
        if (Rank(x) != Rank(y)) return Rank(x) < Rank(y) ? x : y;
        return y.Tie < x.Tie ? y : x;
    }

    /// <summary>§5.3.4, re-derived: (a) leaf, (b) unlocked, (c) scoped, (d) no other waiter, (e) support, (f) slack.</summary>
    private static MaterializeMode RefDonation(PrefixTree t, RadixNode x, int length, bool primary, IWaitingPlanView? waiting)
    {
        PrefixCacheCapabilities caps = t.Caps;
        ScopeRecord rec = t.Scopes[x.ScopeIx];
        bool ok = x.Children.Count == 0
                  && x.LockRef == 0 && x.StateLockRef == 0 && x.PinRef == 0
                  && x.ScopeIx != 0
                  && (primary ? caps.PrimaryResident : caps.EndState is EndStateSupport.DonateOnly or EndStateSupport.CopyAndDonate)
                  && x.Depth - length <= t.Options.DonateTruncateSlackTokens
                  && (rec.WaitingRequests <= 1 || (waiting is not null && rec.WaitingRequests - 1 <= 128 && waiting.NoOtherWaiterTargets(x.ScopeIx, x, t.Version)));
        if (primary)
            return ok ? MaterializeMode.KeepPrimary
                 : caps.AdoptPrimaryOnDisplacement && caps.EndState == EndStateSupport.CopyAndDonate ? MaterializeMode.ConvertPrimaryThenClone : MaterializeMode.None;
        return ok ? MaterializeMode.DonateEndState
             : caps.EndState == EndStateSupport.CopyAndDonate ? MaterializeMode.CloneEndState : MaterializeMode.None;
    }
}
