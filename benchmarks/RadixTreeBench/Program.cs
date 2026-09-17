// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Diagnostics;
using System.Globalization;
using System.Runtime;
using System.Text.Json;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace RadixTreeBench;

/// <summary>
/// RadixTreeBench (DESIGN §13.2, gate BG-15). Synthetic, CPU-only traces over <see cref="PrefixTree"/>:
/// <list type="bullet">
/// <item><c>sessions</c>: 1k sessions × 50 turns growing to 8k tokens, round-robin, 4 shared 512-token system prompts, a regenerate every 10th turn, a byte budget for 800 sessions.</item>
/// <item><c>long32k</c>: 64 sessions × 20 turns at 30k-32k tokens (probe and insert gates at 32k).</item>
/// <item><c>long262k</c>: 8 sessions × 20 turns at 258k-262k tokens (probe gate at 262k).</item>
/// <item><c>queue128</c>: 128 waiting requests, 32 re-matches per step, one admission and one insert per step.</item>
/// <item><c>fanout</c>: one public node with 10k scoped children.</item>
/// <item><c>churn</c>: a budget sized for 64 conversations under 5k new ones (evict-one latency).</item>
/// </list>
/// Usage: <c>dotnet run -c Release --project benchmarks/RadixTreeBench -- --trace all --out rtb.json [--scale 0.1]</c>.
/// Exit code 0 when every BG-15 threshold holds, 1 otherwise.
/// </summary>
internal static class Program
{
    private static readonly string[] AllTraces = { "sessions", "long32k", "long262k", "queue128", "fanout", "churn" };

    private static int Main(string[] args)
    {
        string traces = "all";
        string outPath = null;
        double scale = 1.0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--trace": traces = args[++i]; break;
                case "--out": outPath = args[++i]; break;
                case "--scale": scale = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "-h":
                case "--help":
                    Console.WriteLine("RadixTreeBench --trace all|sessions,long32k,long262k,queue128,fanout,churn [--scale 1.0] [--out rtb.json]");
                    return 0;
                default:
                    Console.Error.WriteLine($"unknown argument {args[i]}");
                    return 2;
            }
        }
        string[] selected = traces == "all" ? AllTraces : traces.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (string t in selected)
            if (!AllTraces.Contains(t)) { Console.Error.WriteLine($"unknown trace {t}"); return 2; }

        Console.WriteLine($"RadixTreeBench: {Environment.ProcessorCount} cores, {RuntimeInformation()}, GC {(GCSettings.IsServerGC ? "server" : "workstation")}/{GCSettings.LatencyMode}, scale {scale}");
        // The allocation gate needs blocking GCs. A background GC ends its mark phase by voiding every thread's
        // allocation context (gc.cpp repair_allocation_contexts(FALSE)) without taking the unused remainder out of
        // the thread's allocated-bytes counter, so GC.GetAllocatedBytesForCurrentThread jumps by up to one
        // allocation quantum (~8 KB) across a window that allocated nothing, whenever that suspension lands in it.
        // The csproj turns concurrent GC off; an environment override (DOTNET_gcConcurrent=1) would bring back a
        // gate that fails at random, so refuse to run rather than report it.
        if (GCSettings.LatencyMode != GCLatencyMode.Batch)
        {
            Console.Error.WriteLine($"RadixTreeBench needs concurrent GC off (GC latency mode is {GCSettings.LatencyMode}, expected Batch): a background GC makes the bytes-per-probe counter jump without an allocation. Unset DOTNET_gcConcurrent / System.GC.Concurrent.");
            return 2;
        }
        // Warm up on small, unrecorded runs of every trace for ~2 s: the first probes of a process pay one-time
        // JIT and type initialisation (latency outliers, and a small allocation on the very first probes), and
        // tiered compilation promotes the hot methods before measurement.
        var warm = Stopwatch.StartNew();
        while (warm.Elapsed.TotalSeconds < 2)
        {
            Bench.Sessions(new Recorder("warmup"), sessions: 40, turns: 12, growth: 150, systemTokens: 512, budgetSessions: 10);
            Bench.Long(new Recorder("warmup"), sessions: 2, turns: 4, startTokens: 4000, growth: 100);
            Bench.Queue(new Recorder("warmup"), queue: 16, steps: 20, promptTokens: 2000, systemTokens: 256);
            Bench.Fanout(new Recorder("warmup"), children: 100, systemTokens: 256, convTokens: 64, probes: 200, quiet: true);
            Bench.Churn(new Recorder("warmup"), sessions: 100, promptTokens: 1000, systemTokens: 256, budgetSessions: 8);
        }
        Thread.Sleep(1000);

        var results = new List<Recorder>();
        foreach (string trace in selected)
        {
            var rec = new Recorder(trace);
            var sw = Stopwatch.StartNew();
            switch (trace)
            {
                case "sessions": Bench.Sessions(rec, sessions: Scale(1000, scale, 20), turns: 50, growth: 150, systemTokens: 512, budgetSessions: Scale(800, scale, 16)); break;
                case "long32k": Bench.Long(rec, sessions: Scale(64, scale, 4), turns: 20, startTokens: 30_000, growth: 100); break;
                case "long262k": Bench.Long(rec, sessions: Scale(8, scale, 2), turns: 20, startTokens: 258_000, growth: 200); break;
                case "queue128": Bench.Queue(rec, queue: 128, steps: Scale(400, scale, 50), promptTokens: 8000, systemTokens: 1024); break;
                case "fanout": Bench.Fanout(rec, children: Scale(10_000, scale, 500), systemTokens: 1024, convTokens: 256, probes: 4000); break;
                case "churn": Bench.Churn(rec, sessions: Scale(5000, scale, 300), promptTokens: 4000, systemTokens: 512, budgetSessions: 64); break;
            }
            rec.WallSeconds = sw.Elapsed.TotalSeconds;
            rec.Print();
            results.Add(rec);
        }

        List<Gate> gates = Gates(results);
        Console.WriteLine();
        Console.WriteLine("BG-15 gates:");
        foreach (Gate g in gates)
            Console.WriteLine($"  {(g.Pass is null ? "n/a " : g.Pass.Value ? "PASS" : "FAIL")}  {g.Name,-40} {g.Measured,14} (threshold {g.Threshold})");
        bool pass = gates.All(g => g.Pass != false);
        Console.WriteLine(pass ? "BG-15: PASS" : "BG-15: FAIL");

        if (outPath is not null)
        {
            var doc = new Dictionary<string, object>
            {
                ["tool"] = "RadixTreeBench",
                ["machine"] = new Dictionary<string, object> { ["cores"] = Environment.ProcessorCount, ["runtime"] = RuntimeInformation(), ["os"] = Environment.OSVersion.ToString() },
                ["scale"] = scale,
                ["traces"] = results.Select(r => r.ToJson()).ToList(),
                ["gates"] = gates.Select(g => new Dictionary<string, object> { ["name"] = g.Name, ["measured"] = g.Measured, ["threshold"] = g.Threshold, ["pass"] = g.Pass }).ToList(),
                ["pass"] = pass,
            };
            File.WriteAllText(outPath, JsonSerializer.Serialize(doc, new JsonSerializerOptions { WriteIndented = true }));
            Console.WriteLine($"wrote {outPath}");
        }
        return pass ? 0 : 1;
    }

    private static int Scale(int n, double scale, int min) => Math.Max(min, (int)Math.Round(n * scale));

    private static string RuntimeInformation() => System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription;

    private sealed record Gate(string Name, string Measured, string Threshold, bool? Pass);

    private static List<Gate> Gates(List<Recorder> results)
    {
        Recorder Find(string name) => results.FirstOrDefault(r => r.Trace == name);
        var gates = new List<Gate>();
        Recorder l32 = Find("long32k"), l262 = Find("long262k"), q = Find("queue128");
        if (l32 is not null)
        {
            gates.Add(Um("probe p99 @ 32k", l32.Probe.P(99), 200));
            gates.Add(Um("insert p99 @ 32k", l32.Insert.P(99), 500));
        }
        if (l262 is not null) gates.Add(Um("probe p99 @ 262k", l262.Probe.P(99), 2000));
        var evict = new Samples();
        foreach (Recorder r in results) evict.AddAll(r.EvictOne);
        if (evict.Count > 0) gates.Add(Um("evict one victim p99", evict.P(99), 50));
        if (q is not null) gates.Add(new Gate("admission overhead per step p99 @ queue 128", $"{q.Step.P(99) / 1000.0:F3} ms", "5 ms", q.Step.P(99) <= 5000));
        long bytes = results.Sum(r => r.ProbeAllocatedBytes);
        long probes = results.Sum(r => r.ProbesMeasuredForAllocation);
        if (probes > 0)
        {
            double perProbe = (double)bytes / probes;
            gates.Add(new Gate("bytes allocated per probe", $"{perProbe:F3} B", "0", bytes == 0));
        }
        return gates;
    }

    private static Gate Um(string name, double measuredMicros, double thresholdMicros) =>
        new(name, $"{measuredMicros:F1} µs", $"{thresholdMicros} µs", measuredMicros <= thresholdMicros);
}

/// <summary>Latency samples in Stopwatch ticks, reported in microseconds.</summary>
internal sealed class Samples
{
    private long[] _data = new long[1024];
    internal int Count { get; private set; }

    internal void Add(long ticks)
    {
        if (Count == _data.Length) Array.Resize(ref _data, _data.Length * 2);
        _data[Count++] = ticks;
    }

    internal void AddAll(Samples other)
    {
        for (int i = 0; i < other.Count; i++) Add(other._data[i]);
    }

    internal double P(double percentile)
    {
        if (Count == 0) return 0;
        long[] sorted = _data.AsSpan(0, Count).ToArray();
        Array.Sort(sorted);
        int ix = (int)Math.Ceiling(percentile / 100.0 * Count) - 1;
        return ToMicros(sorted[Math.Clamp(ix, 0, Count - 1)]);
    }

    internal double Mean => Count == 0 ? 0 : ToMicros((long)_data.AsSpan(0, Count).ToArray().Average());

    internal static double ToMicros(long ticks) => ticks * 1_000_000.0 / Stopwatch.Frequency;

    internal Dictionary<string, object> ToJson() => new()
    {
        ["count"] = Count,
        ["mean_us"] = Math.Round(Mean, 3),
        ["p50_us"] = Math.Round(P(50), 3),
        ["p95_us"] = Math.Round(P(95), 3),
        ["p99_us"] = Math.Round(P(99), 3),
        ["max_us"] = Math.Round(P(100), 3),
    };
}

internal sealed class Recorder
{
    internal Recorder(string trace) { Trace = trace; }

    internal string Trace { get; }
    internal readonly Samples Probe = new(), Acquire = new(), Insert = new(), EvictOne = new(), Step = new(), KeyBuild = new();
    internal long ProbeAllocatedBytes, ProbesMeasuredForAllocation;
    internal readonly List<long> ProbeAllocations = new();   // bytes per probe, in order
    /// <summary>Bytes allocated by probes in the second half of the trace (after reusable buffers have grown).</summary>
    internal long SteadyStateProbeBytes => ProbeAllocations.Skip(ProbeAllocations.Count / 2).Sum();
    internal int ProbesThatAllocated => ProbeAllocations.Count(b => b > 0);
    internal long Admissions, Reused, Prompted, Evictions, ScanSkips, EvictCalls, RopeCompactions, MaxNodes, Splits;
    internal int MinPromptTokens = int.MaxValue, MaxPromptTokens;
    internal double WallSeconds;

    internal void NotePrompt(int tokens)
    {
        MinPromptTokens = Math.Min(MinPromptTokens, tokens);
        MaxPromptTokens = Math.Max(MaxPromptTokens, tokens);
    }

    internal void Print()
    {
        Console.WriteLine();
        Console.WriteLine($"[{Trace}] {WallSeconds:F1} s, {Admissions} admissions, prompts {MinPromptTokens}-{MaxPromptTokens} tokens, reuse {(Prompted == 0 ? 0 : 100.0 * Reused / Prompted):F1}%, max nodes {MaxNodes}, evictions {Evictions} (scan skips {ScanSkips} over {EvictCalls} evict calls), splits {Splits}, rope compactions {RopeCompactions}");
        Line("probe (Match+Evaluate)", Probe);
        Line("acquire", Acquire);
        Line("insert+attach", Insert);
        Line("evict one victim", EvictOne);
        Line("admission step", Step);
        Line("key build", KeyBuild);
        if (ProbesMeasuredForAllocation > 0)
            Console.WriteLine($"  bytes/probe {(double)ProbeAllocatedBytes / ProbesMeasuredForAllocation:F3} over {ProbesMeasuredForAllocation} probes ({ProbesThatAllocated} probes allocated, max {(ProbeAllocations.Count == 0 ? 0 : ProbeAllocations.Max())} B; second half {SteadyStateProbeBytes} B)");
    }

    private static void Line(string name, Samples s)
    {
        if (s.Count == 0) return;
        Console.WriteLine($"  {name,-24} n={s.Count,8}  p50 {s.P(50),10:F1} µs  p95 {s.P(95),10:F1} µs  p99 {s.P(99),10:F1} µs  max {s.P(100),10:F1} µs");
    }

    internal Dictionary<string, object> ToJson() => new()
    {
        ["trace"] = Trace,
        ["wall_s"] = Math.Round(WallSeconds, 2),
        ["admissions"] = Admissions,
        ["prompt_tokens_min"] = MinPromptTokens == int.MaxValue ? 0 : MinPromptTokens,
        ["prompt_tokens_max"] = MaxPromptTokens,
        ["reuse_percent"] = Prompted == 0 ? 0 : Math.Round(100.0 * Reused / Prompted, 2),
        ["max_nodes"] = MaxNodes,
        ["evictions"] = Evictions,
        ["evict_scan_skips"] = ScanSkips,
        ["evict_calls"] = EvictCalls,
        ["evict_scan_skips_per_call"] = EvictCalls == 0 ? 0 : (double)ScanSkips / EvictCalls,
        ["splits"] = Splits,
        ["rope_compactions"] = RopeCompactions,
        ["bytes_per_probe"] = ProbesMeasuredForAllocation == 0 ? 0 : (double)ProbeAllocatedBytes / ProbesMeasuredForAllocation,
        ["probe_allocated_bytes_total"] = ProbeAllocatedBytes,
        ["probes_that_allocated"] = ProbesThatAllocated,
        ["probe"] = Probe.ToJson(),
        ["acquire"] = Acquire.ToJson(),
        ["insert"] = Insert.ToJson(),
        ["evict_one"] = EvictOne.ToJson(),
        ["admission_step"] = Step.ToJson(),
        ["key_build"] = KeyBuild.ToJson(),
    };
}

internal static class Bench
{
    private const int BlockSize = 256;
    private const long BytesPerToken = 100;

    private static PrefixCacheCapabilities Caps() => new()
    {
        Class = FamilyClass.R,
        Readiness = PrefixCacheMode.Legacy,
        NamespaceFingerprint = "radix-tree-bench",
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true,
        PrimaryResident = false,
        Truncation = TruncationKind.Any,
        RewindCapTokens = 16,
        Pages = PageSupport.None,
        ReuseAcrossMediaSpan = true,
    };

    private static PrefixTree NewTree(long hostBudgetBytes) => new(new PrefixTreeOptions
    {
        Capabilities = Caps(),
        BlockSize = BlockSize,
        PublicMax = 4,
        ScopedEndStateLeavesMax = 0,
        OptionCapBytes = new ResourceVector { HostKv = hostBudgetBytes },
        TrackReceipts = false,
        StrictReceipts = false,
        ClockMs = () => 0,
    });

    /// <summary>A deterministic pseudo-random token stream per (stream, position).</summary>
    private static int Token(long stream, int position)
    {
        ulong x = (ulong)stream * 0x9E3779B97F4A7C15UL + (ulong)position * 0xBF58476D1CE4E5B9UL;
        x ^= x >> 31; x *= 0x94D049BB133111EBUL; x ^= x >> 29;
        return (int)(x % 150_000);
    }

    private static void Fill(int[] buffer, int from, int to, long stream)
    {
        for (int i = from; i < to; i++) buffer[i] = Token(stream, i);
    }

    private sealed class Conversation
    {
        public int ScopeIx;
        public int[] Tokens;
        public int Length;           // tokens so far (prompt + output of previous turns)
        public int P;
    }

    private static KeyRope BuildKey(Recorder rec, PrefixTree tree, int[] tokens, int length)
    {
        var segment = new ArraySegment<int>(tokens, 0, length);
        long t0 = Stopwatch.GetTimestamp();
        KeyRope key = RadixKeyBuilder.BuildPrompt(segment, Array.Empty<MediaSpanRecord>(), tree.KeyPool);
        rec.KeyBuild.Add(Stopwatch.GetTimestamp() - t0);
        return key;
    }

    private static MatchPlan Probe(Recorder rec, PrefixTree tree, in MatchRequest r, MatchPlan plan)
    {
        long a0 = GC.GetAllocatedBytesForCurrentThread();
        long t0 = Stopwatch.GetTimestamp();
        tree.Plan(r, plan);
        long dt = Stopwatch.GetTimestamp() - t0;
        long allocated = GC.GetAllocatedBytesForCurrentThread() - a0;
        rec.Probe.Add(dt);
        rec.ProbeAllocatedBytes += allocated;
        rec.ProbeAllocations.Add(allocated);
        rec.ProbesMeasuredForAllocation++;
        return plan;
    }

    /// <summary>One admission: probe, acquire, materialize, run (nothing), finish with an insert, release, evict to budget.</summary>
    private static void Turn(Recorder rec, PrefixTree tree, Conversation c, int promptTokens, int outputTokens, long stream, MatchPlan plan)
    {
        if (c.Tokens.Length < promptTokens + outputTokens)
            Array.Resize(ref c.Tokens, Math.Max(promptTokens + outputTokens, c.Tokens.Length * 2));
        Fill(c.Tokens, c.Length, promptTokens, stream);
        KeyRope key = BuildKey(rec, tree, c.Tokens, promptTokens);
        rec.NotePrompt(promptTokens);
        tree.NoteRequests(c.ScopeIx, +1, 0);
        var r = new MatchRequest(key, promptTokens, c.ScopeIx, c.P, promptTokens - 1, Array.Empty<MediaSpanRecord>(), ExpectedRoute.PerSequenceFused, false);
        Probe(rec, tree, r, plan);
        long t0 = Stopwatch.GetTimestamp();
        LockReceipt receipt = tree.Acquire(plan);
        if (plan.Mode == MaterializeMode.DonateEndState)
        {
            tree.MarkDonationPending(plan.PayloadNode);
            tree.CommitDonation(plan.PayloadNode);
            tree.ReleaseState(ref receipt);
        }
        else if (plan.Mode == MaterializeMode.CloneEndState)
        {
            tree.ReleaseState(ref receipt);
        }
        rec.Acquire.Add(Stopwatch.GetTimestamp() - t0);
        tree.NoteRequests(c.ScopeIx, -1, +1);
        rec.Admissions++;
        rec.Prompted += promptTokens;
        rec.Reused += plan.Length;

        // Finish: output tokens, insert, attach the donated holder.
        int total = promptTokens + outputTokens;
        Fill(c.Tokens, promptTokens, total, stream + 1);
        RadixKeyBuilder.AppendOutput(key, promptTokens, new ReadOnlySpan<int>(c.Tokens, promptTokens, outputTokens), tree.KeyPool);
        t0 = Stopwatch.GetTimestamp();
        RadixNode node = tree.Insert(key, total, c.ScopeIx, c.P, NodeFlags.None, null);
        if (node.EndState is null)
        {
            tree.AttachEndState(node, new EndStatePayload
            {
                Key = tree.MintKey(),
                Kind = EndStateKind.Holder,
                Origin = PayloadOrigin.Donation,
                Footprint = new PayloadFootprint(total, total, new ResourceVector { HostKv = total * BytesPerToken }, 0),
            });
        }
        rec.Insert.Add(Stopwatch.GetTimestamp() - t0);
        tree.Release(ref receipt);
        tree.CollectIfEmpty(node);
        tree.NoteRequests(c.ScopeIx, 0, -1);
        tree.ReleaseRopeOwner(key);
        c.Length = total;
        EvictToBudget(rec, tree);
        rec.MaxNodes = Math.Max(rec.MaxNodes, tree.NodeCount);
    }

    private static void EvictToBudget(Recorder rec, PrefixTree tree)
    {
        long victims = tree.Counters.Evictions;
        long t0 = Stopwatch.GetTimestamp();
        long cap = tree.EffectiveCap(ResourceClass.HostKv);
        if (tree.Cached.HostKv > cap)
            tree.Evict(ResourceClass.HostKv, tree.Cached.HostKv - cap, ReleaseReason.Evicted, EvictionTier.ScopeNewest);
        long dt = Stopwatch.GetTimestamp() - t0;
        long evicted = tree.Counters.Evictions - victims;
        if (evicted > 0)
        {
            // Per-victim latency: the call divided evenly over the victims it took (model release excluded).
            for (long i = 0; i < evicted; i++) rec.EvictOne.Add(dt / evicted);
            rec.Evictions += evicted;
        }
        if (tree.Reclaim.Count >= 64) tree.DrainReclaimQueue(null);
        rec.ScanSkips = tree.Counters.EvictScanSkips;
        rec.EvictCalls = tree.Counters.EvictCalls;
        rec.RopeCompactions = tree.Counters.RopeCompactions;
        rec.Splits = tree.Counters.Splits;
    }

    private static Conversation NewConversation(PrefixTree tree, long stream, int systemTokens, int systemId, int capacity)
    {
        var c = new Conversation
        {
            ScopeIx = tree.InternScope(new ScopeId(new UInt128((ulong)stream + 1, 0x5eed)), ScopeKind.Session),
            Tokens = new int[capacity],
            P = systemTokens,
        };
        Fill(c.Tokens, 0, systemTokens, 10_000_000 + systemId);
        c.Length = systemTokens;
        return c;
    }

    internal static void Sessions(Recorder rec, int sessions, int turns, int growth, int systemTokens, int budgetSessions)
    {
        int finalTokens = systemTokens + turns * growth;
        PrefixTree tree = NewTree(budgetSessions * (long)finalTokens * BytesPerToken);
        var plan = new MatchPlan();
        var convs = new Conversation[sessions];
        for (int s = 0; s < sessions; s++) convs[s] = NewConversation(tree, s, systemTokens, s % 4, finalTokens + 64);
        int user = growth * 2 / 3, output = growth - user;
        for (int t = 0; t < turns; t++)
            for (int s = 0; s < sessions; s++)
            {
                Conversation c = convs[s];
                // Every 10th turn regenerates: the history loses its last 50 tokens (a fork past the rewind
                // cap, so the tree splits instead of truncating).
                if (t % 10 == 9 && c.Length > systemTokens + 50) c.Length -= 50;
                // The first turn of a session publishes its system prompt as a public boundary.
                if (t == 0) PublishSystemPrompt(tree, c);
                Turn(rec, tree, c, c.Length + user, output, ((long)s << 20) + t * 2 + 1_000_000, plan);
            }
        tree.DrainReclaimQueue(null);
    }

    private static void PublishSystemPrompt(PrefixTree tree, Conversation c)
    {
        KeyRope key = RadixKeyBuilder.BuildPrompt(new ArraySegment<int>(c.Tokens, 0, c.P), Array.Empty<MediaSpanRecord>(), tree.KeyPool);
        RadixNode node = tree.Insert(key, c.P, 0, c.P, NodeFlags.None, null);
        if (node.EndState is null)
            tree.AttachEndState(node, new EndStatePayload { Key = tree.MintKey(), Footprint = new PayloadFootprint(c.P, c.P, new ResourceVector { HostKv = c.P * BytesPerToken }, 0) });
        tree.ReleaseRopeOwner(key);
    }

    internal static void Long(Recorder rec, int sessions, int turns, int startTokens, int growth)
    {
        PrefixTree tree = NewTree(long.MaxValue / 4);
        var plan = new MatchPlan();
        int user = growth / 2, output = growth - user;
        for (int s = 0; s < sessions; s++)
        {
            Conversation c = NewConversation(tree, 1_000_000 + s, 1024, s % 2, startTokens + turns * growth + 64);
            Fill(c.Tokens, c.Length, startTokens - user, 50_000_000 + s);   // the long document
            c.Length = startTokens - user;
            for (int t = 0; t < turns; t++)
                Turn(rec, tree, c, c.Length + user, output, ((long)(s + 7000) << 20) + t * 2, plan);
            // Free the session between runs: the tree keeps at most one long conversation resident.
            tree.RetireScope(tree.Scopes[c.ScopeIx].Id);
            tree.DrainReclaimQueue(null);
        }
    }

    internal static void Queue(Recorder rec, int queue, int steps, int promptTokens, int systemTokens)
    {
        PrefixTree tree = NewTree(64L * promptTokens * BytesPerToken);
        var waiting = new List<(Conversation Conv, KeyRope Key, MatchPlan Plan, int Tokens)>();
        long stream = 90_000_000;
        void AddWaiting()
        {
            Conversation c = NewConversation(tree, stream++, systemTokens, 0, promptTokens + 64);
            Fill(c.Tokens, c.Length, promptTokens, stream * 3);
            KeyRope key = RadixKeyBuilder.BuildPrompt(new ArraySegment<int>(c.Tokens, 0, promptTokens), Array.Empty<MediaSpanRecord>(), tree.KeyPool);
            tree.NoteRequests(c.ScopeIx, +1, 0);
            rec.NotePrompt(promptTokens);
            waiting.Add((c, key, new MatchPlan(), promptTokens));
        }
        for (int i = 0; i < queue; i++) AddWaiting();
        // A background conversation that keeps publishing (the tree changes every step).
        Conversation bg = NewConversation(tree, 1, systemTokens, 0, systemTokens + steps * 200 + 1024);
        PublishSystemPrompt(tree, bg);
        var bgPlan = new MatchPlan();
        for (int step = 0; step < steps; step++)
        {
            long t0 = Stopwatch.GetTimestamp();
            int rematches = 0;
            int admit = -1;
            for (int i = 0; i < waiting.Count; i++)
            {
                var w = waiting[i];
                if (w.Plan.Version != tree.Version)
                {
                    if (rematches >= 32) continue;                   // stale beyond the budget: not admitted this step
                    var r = new MatchRequest(w.Key, w.Tokens, w.Conv.ScopeIx, systemTokens, w.Tokens - 1, Array.Empty<MediaSpanRecord>(), ExpectedRoute.BatchedPaged, false);
                    Probe(rec, tree, r, w.Plan);
                    rematches++;
                }
                if (admit < 0 && w.Plan.Version == tree.Version) admit = i;
            }
            if (admit >= 0)
            {
                var w = waiting[admit];
                LockReceipt receipt = tree.Acquire(w.Plan);
                if (w.Plan.Mode is MaterializeMode.CloneEndState or MaterializeMode.DonateEndState) tree.ReleaseState(ref receipt);
                tree.NoteRequests(w.Conv.ScopeIx, -1, +1);
                rec.Admissions++;
                rec.Prompted += w.Tokens;
                rec.Reused += w.Plan.Length;
                // It finishes at once: insert its prompt, release, and a new request joins the queue.
                RadixNode node = tree.Insert(w.Key, w.Tokens, w.Conv.ScopeIx, systemTokens, NodeFlags.None, null);
                if (node.EndState is null)
                    tree.AttachEndState(node, new EndStatePayload { Key = tree.MintKey(), Footprint = new PayloadFootprint(w.Tokens, w.Tokens, new ResourceVector { HostKv = w.Tokens * BytesPerToken }, 0) });
                tree.Release(ref receipt);
                tree.NoteRequests(w.Conv.ScopeIx, 0, -1);
                tree.ReleaseRopeOwner(w.Key);
                waiting.RemoveAt(admit);
                EvictToBudget(rec, tree);
            }
            rec.Step.Add(Stopwatch.GetTimestamp() - t0);
            if (admit >= 0) AddWaiting();
            Turn(new Recorder("bg"), tree, bg, bg.Length + 150, 50, 777_000 + step * 2, bgPlan);
            rec.MaxNodes = Math.Max(rec.MaxNodes, tree.NodeCount);
        }
    }

    internal static void Fanout(Recorder rec, int children, int systemTokens, int convTokens, int probes, bool quiet = false)
    {
        PrefixTree tree = NewTree(long.MaxValue / 4);
        var plan = new MatchPlan();
        var convs = new Conversation[children];
        var rng = new Random(42);
        for (int s = 0; s < children; s++)
        {
            convs[s] = NewConversation(tree, 200_000_000 + s, systemTokens, 0, systemTokens + convTokens * 2 + 64);
            Turn(rec, tree, convs[s], systemTokens + convTokens, 16, 300_000_000L + s, plan);
        }
        // Publish the public boundary once.
        RadixNode pub = tree.Insert(RadixKeyBuilder.BuildPrompt(new ArraySegment<int>(convs[0].Tokens, 0, systemTokens), Array.Empty<MediaSpanRecord>(), tree.KeyPool),
                                    systemTokens, 0, systemTokens, NodeFlags.None, null);
        if (pub.EndState is null)
            tree.AttachEndState(pub, new EndStatePayload { Key = tree.MintKey(), Footprint = new PayloadFootprint(systemTokens, systemTokens, new ResourceVector { HostKv = systemTokens * BytesPerToken }, 0) });
        // Probe only (no mutation): existing scopes (a deep own hit) and new scopes (public only).
        rec.Probe.AddAll(new Samples());
        var probeRec = new Recorder("fanout-probes");
        for (int i = 0; i < probes; i++)
        {
            Conversation c = convs[rng.Next(children)];
            bool own = i % 2 == 0;
            int length = c.Length + 8;
            int[] tokens = c.Tokens;
            Fill(tokens, c.Length, length, 400_000_000L + i);
            KeyRope key = RadixKeyBuilder.BuildPrompt(new ArraySegment<int>(tokens, 0, length), Array.Empty<MediaSpanRecord>(), tree.KeyPool);
            int scopeIx = own ? c.ScopeIx : 0;
            var r = new MatchRequest(key, length, scopeIx, systemTokens, length - 1, Array.Empty<MediaSpanRecord>(), ExpectedRoute.PerSequenceFused, false);
            Probe(probeRec, tree, r, plan);
            probeRec.Admissions++;
            probeRec.Prompted += length;
            probeRec.Reused += plan.Length;
            key.ReturnChunks(tree.KeyPool);
        }
        rec.Probe.AddAll(probeRec.Probe);
        rec.ProbeAllocatedBytes += probeRec.ProbeAllocatedBytes;
        rec.ProbesMeasuredForAllocation += probeRec.ProbesMeasuredForAllocation;
        rec.Reused = probeRec.Reused;
        rec.Prompted = probeRec.Prompted;
        rec.MaxNodes = tree.NodeCount;
        if (!quiet) Console.WriteLine($"  fanout: public node children = {pub.Children.Count}");
    }

    internal static void Churn(Recorder rec, int sessions, int promptTokens, int systemTokens, int budgetSessions)
    {
        PrefixTree tree = NewTree(budgetSessions * (long)(promptTokens + 64) * BytesPerToken);
        var plan = new MatchPlan();
        for (int s = 0; s < sessions; s++)
        {
            Conversation c = NewConversation(tree, 600_000_000 + s, systemTokens, s % 3, promptTokens + 128);
            Turn(rec, tree, c, promptTokens, 64, 700_000_000L + s, plan);
        }
        tree.DrainReclaimQueue(null);
    }
}
