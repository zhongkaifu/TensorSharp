// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// --bench-workload / --bench-suite (TensorSharp.Cli/BenchmarkSuite.cs) and the cross-node eviction it drives
// (ModelBase.EvictHostTablePageCache): many benchmark rows in one process, each through the --benchmark loop, a cold
// row evicting the model's host-mapped tables on every node first. No GPU: a CPU stand-in model whose logits depend
// on the sequence, and whose one "host-mapped table" is a range of a file it maps and reads on every forward.
using System.Collections.Concurrent;
using System.IO.MemoryMappedFiles;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class BenchmarkSuiteTests : IDisposable
{
    private const int Vocab = 128;
    private readonly string _dir = Path.Combine(AppContext.BaseDirectory, "bench-suite-" + Guid.NewGuid().ToString("N"));
    private readonly string _gguf;

    public BenchmarkSuiteTests()
    {
        Directory.CreateDirectory(_dir);
        _gguf = Path.Combine(_dir, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_gguf);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private static readonly BenchWorkloadSpec.Defaults Defaults = new() { Prefill = 32, Decode = 64, Runs = 1, Mode = "greedy" };

    // ---------------------------------------------------------------- specs

    [Fact]
    public void Parse_ReadsEveryKey_AndBareWords()
    {
        var w = BenchWorkloadSpec.Parse(" label=pp2048-cold , prefill=2048,decode=128,runs=4,mode=random,cold,seed=7,check=0 ", Defaults);
        Assert.Equal("pp2048-cold", w.Label);
        Assert.Equal(2048, w.Prefill);
        Assert.Equal(128, w.Decode);
        Assert.Equal(4, w.Runs);
        Assert.Equal("random", w.Mode);
        Assert.Equal(BenchEvict.Row, w.Evict);
        Assert.Equal(7, w.Seed);
        Assert.False(w.Check);
        Assert.Equal("synthetic", w.Source);

        var t = BenchWorkloadSpec.Parse("text=/root/s7/r8k.txt,prefill=8000,decode=0,evict=run,chunked", Defaults);
        Assert.Equal("text", t.Source);
        Assert.Equal("/root/s7/r8k.txt", t.TextPath);
        Assert.Equal(8000, t.Prefill);
        Assert.Equal(0, t.Decode);
        Assert.Equal(BenchEvict.Run, t.Evict);
        Assert.True(t.Chunked);
    }

    [Fact]
    public void Parse_TakesTheProcessDefaults_ForUnsetFields()
    {
        var d = new BenchWorkloadSpec.Defaults { Prefill = 512, Decode = 128, Runs = 4, Chunked = true, Mode = "random" };
        var w = BenchWorkloadSpec.Parse("label=x", d);
        Assert.Equal((512, 128, 4, true, "random", BenchEvict.None), (w.Prefill.Value, w.Decode, w.Runs, w.Chunked, w.Mode, w.Evict));
        Assert.Equal(BenchWorkloadSpec.DefaultSeed, w.Seed);
        Assert.True(w.Check);
        // A real prompt has no default length: the whole file.
        Assert.Null(BenchWorkloadSpec.Parse("tokens=t.json", d).Prefill);
    }

    [Theory]
    [InlineData("", "empty")]
    [InlineData("prefill=0", "prefill must be an integer >= 1")]
    [InlineData("decode=-1", "decode must be an integer >= 0")]
    [InlineData("runs=0", "runs must be an integer >= 1")]
    [InlineData("mode=beam", "mode must be greedy, fixed or random")]
    [InlineData("evict=always", "evict must be none, row or run")]
    [InlineData("hot", "bare words")]
    [InlineData("prefill=8,prefill=9", "given twice")]
    [InlineData("text=a.txt,tokens=b.json", "not both")]
    [InlineData("label=has space", "label")]
    [InlineData("speed=fast", "unknown key")]
    [InlineData("check=maybe", "check must be 0 or 1")]
    public void Parse_RefusesBadSpecs(string spec, string why)
        => Assert.Contains(why, Assert.Throws<ArgumentException>(() => BenchWorkloadSpec.Parse(spec, Defaults)).Message);

    [Fact]
    public void Labels_AreDerivedAndUnique_AndExplicitDuplicatesAreRefused()
    {
        var rows = new[] { "prefill=2048,decode=128,cold", "prefill=2048,decode=128", "prefill=2048,decode=128",
                "text=/x/r8k.txt,prefill=4000,decode=8", "label=mine,evict=run" }
            .Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        BenchWorkloadSpec.AssignLabels(rows);
        Assert.Equal(new[] { "pp2048-d128-cold", "pp2048-d128-warm", "pp2048-d128-warm.2", "r8k-4000-d8-warm", "mine" },
            rows.Select(r => r.Label));

        var dup = new[] { "label=a", "label=a,decode=3" }.Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        Assert.Contains("share the label 'a'", Assert.Throws<ArgumentException>(() => BenchWorkloadSpec.AssignLabels(dup)).Message);
    }

    [Fact]
    public void SuiteFile_ReadsLinesOrAJsonArray()
    {
        string text = Path.Combine(_dir, "rows.txt");
        File.WriteAllText(text, "# cold first\nlabel=a,prefill=8,cold\n\n  label=b,prefill=8  \n");
        Assert.Equal(new[] { "label=a,prefill=8,cold", "label=b,prefill=8" }, BenchWorkloadSpec.ReadSuiteFile(text));

        string json = Path.Combine(_dir, "rows.json");
        File.WriteAllText(json, """[{"label":"a","prefill":2048,"cold":true,"check":false},{"text":"/r/r2k.txt","decode":4}]""");
        var specs = BenchWorkloadSpec.ReadSuiteFile(json);
        Assert.Equal(new[] { "label=a,prefill=2048,cold=1,check=0", "text=/r/r2k.txt,decode=4" }, specs);
        // "cold=1" is the key spelling of the bare word.
        Assert.Equal(BenchEvict.Row, BenchWorkloadSpec.Parse(specs[0], Defaults).Evict);
        Assert.Equal(BenchEvict.None, BenchWorkloadSpec.Parse("cold=0", Defaults).Evict);
        Assert.Equal(BenchEvict.None, BenchWorkloadSpec.Parse("warm=1", Defaults).Evict);
        Assert.Contains("given twice", Assert.Throws<ArgumentException>(() => BenchWorkloadSpec.Parse("cold=1,evict=run", Defaults)).Message);
    }

    [Fact]
    public void Preflight_RefusesAMissingPromptFile_AndAColdRowWhereNothingCanBeEvicted()
    {
        var missing = new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=t,text=" + Path.Combine(_dir, "nope.txt"), Defaults) };
        Assert.Contains("does not exist", Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.PreflightBenchmarkSuite(missing)).Message);

        var cold = new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=c,prefill=8,cold", Defaults) };
        if (OperatingSystem.IsLinux())
            TensorSharp.Cli.Program.PreflightBenchmarkSuite(cold);
        else
            Assert.Contains("not supported on this OS",
                Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.PreflightBenchmarkSuite(cold)).Message);
        TensorSharp.Cli.Program.PreflightBenchmarkSuite(new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=w,prefill=8", Defaults) });
    }

    /// <summary>The DeepSeek V4.1 perf rows the VM blocks run (eng/validation/suites) parse, with unique labels, and
    /// MainCore reads the flags and runs the suite (it is one long method with no seam).</summary>
    [Fact]
    public void ShippedSuiteFile_Parses_AndMainCoreRunsTheSuite()
    {
        string root = new DirectoryInfo(AppContext.BaseDirectory).FullName;
        while (!Directory.Exists(Path.Combine(root, "TensorSharp.Runtime")))
            root = Directory.GetParent(root)?.FullName ?? throw new DirectoryNotFoundException("no repository root");
        var rows = BenchWorkloadSpec.ReadSuiteFile(Path.Combine(root, "eng", "validation", "suites", "dsv41-1017-perf.txt"))
            .Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        BenchWorkloadSpec.AssignLabels(rows);
        Assert.True(rows.Count >= 10);
        Assert.Equal(rows.Count, rows.Select(r => r.Label).Distinct().Count());
        Assert.Equal(BenchEvict.Row, rows.First(r => r.Label == "pp2048-cold").Evict);
        Assert.Equal(BenchEvict.None, rows.First(r => r.Label == "pp2048-warm").Evict);
        Assert.All(rows.Where(r => r.Source == "synthetic"), r => Assert.Equal(BenchWorkloadSpec.DefaultSeed, r.Seed));

        string source = File.ReadAllText(Path.Combine(root, "TensorSharp.Cli", "Program.cs"));
        foreach (string expected in new[]
        {
            "case \"--bench-workload\": benchWorkloadSpecs.Add(args[++i]);",
            "case \"--bench-suite\": benchWorkloadSpecs.AddRange(BenchWorkloadSpec.ReadSuiteFile(args[++i]));",
            "PreflightBenchmarkSuite(benchSuite);",
            "RunBenchmarkSuite(model, PrepareBenchmarkSuite(model, benchSuite), new BenchSuiteOptions",
        })
            Assert.Contains(expected, source, StringComparison.Ordinal);
    }

    [Fact]
    public void MeasureFlag_Parses()
    {
        Assert.Equal(HostPageCacheMeasure.Sampled, BenchSuiteOptions.ParseMeasure("sampled"));
        Assert.Equal(HostPageCacheMeasure.Full, BenchSuiteOptions.ParseMeasure("FULL"));
        Assert.Equal(HostPageCacheMeasure.None, BenchSuiteOptions.ParseMeasure("none"));
        Assert.Throws<ArgumentException>(() => BenchSuiteOptions.ParseMeasure("most"));
    }

    [Fact]
    public void Usage_DocumentsTheSuiteFlags()
    {
        var documented = CliUsage.DocumentedFlags().ToHashSet(StringComparer.Ordinal);
        foreach (string flag in new[] { "--bench-workload", "--bench-suite", "--bench-json", "--bench-evict-measure", "--bench-evict-strict" })
            Assert.Contains(flag, documented);
    }

    // ---------------------------------------------------------------- the suite on a model

    /// <summary>Every row runs the --benchmark loop: a random row with the default seed replays exactly what a
    /// one-row --benchmark --bench-random-tokens process feeds, a second row with the same seed replays the first
    /// (the warm-after-cold method), and a text / token-file row prefills that file's tokens. The JSON summary holds
    /// every run of every row.</summary>
    [Fact]
    public void Suite_RunsEveryRow_ThroughTheBenchmarkLoop_AndWritesTheSummary()
    {
        string text = Path.Combine(_dir, "prompt.txt");
        File.WriteAllText(text, "Paris is the capital of France.");
        string tokens = Path.Combine(_dir, "prompt.json");
        File.WriteAllText(tokens, "[40, 41, 42, 43, 44, 45, 46]");
        string json = Path.Combine(_dir, "out", "suite.json");

        using var model = new SuiteModel(_gguf, tablePath: null);
        TensorSharp.Cli.Program.RunBenchmark(model, prefillTokens: 12, decodeTokens: 3, runs: 2, randomTokens: true);
        int[][] oneRowPrompts = model.Prompts.ToArray();
        model.Prompts.Clear();

        var rows = new[]
        {
            "label=a,prefill=12,decode=3,runs=2,mode=random",
            "label=b,prefill=12,decode=3,runs=2,mode=random",
            "label=c,prefill=12,decode=3,runs=2,mode=random,seed=5,check=0",
            "label=t," + "text=" + text + ",decode=2,runs=1",
            "label=k,tokens=" + tokens + ",prefill=5,decode=2,runs=1,mode=fixed",
            "label=g,prefill=9,decode=0,runs=3",
        }.Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        TensorSharp.Cli.Program.PreflightBenchmarkSuite(rows);
        var prepared = TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, rows);
        TensorSharp.Cli.Program.RunBenchmarkSuite(model, prepared, new BenchSuiteOptions { JsonPath = json, EvictMeasure = HostPageCacheMeasure.Full, Backend = "cpu", ModelPath = _gguf });

        // Row a = the one-row process (2 timed prompts + the untimed check's prompt), row b = row a, row c differs.
        int[][] p = model.Prompts.ToArray();
        Assert.Equal(oneRowPrompts, p.Take(3));
        Assert.Equal(p.Take(3), p.Skip(3).Take(3));
        Assert.NotEqual(p[0], p[6]);
        Assert.Equal(2, p.Skip(6).Take(2).Count());           // row c: no untimed check
        Assert.Equal("Paris is the capital of France.".Select(c => (int)c), p[8]);
        Assert.Equal(new[] { 40, 41, 42, 43, 44 }, p[9]);       // capped at prefill=5; then the untimed check's prompt
        Assert.Equal(new[] { 40, 41, 42, 43, 44 }, p[10]);
        Assert.Equal(3, p.Skip(11).Count());                    // row g: three prefill-only runs

        using var doc = JsonDocument.Parse(File.ReadAllText(json));
        JsonElement root = doc.RootElement;
        Assert.Equal("tensorsharp.bench-suite/1", root.GetProperty("schema").GetString());
        Assert.Equal("completed", root.GetProperty("status").GetString());
        Assert.Equal("full", root.GetProperty("evict_measure").GetString());
        var w = root.GetProperty("workloads").EnumerateArray().ToArray();
        Assert.Equal(new[] { "a", "b", "c", "t", "k", "g" }, w.Select(x => x.GetProperty("label").GetString()));
        Assert.Equal(new[] { 2, 2, 2, 1, 1, 3 }, w.Select(x => x.GetProperty("samples").GetArrayLength()));
        Assert.Equal("random-inference", w[0].GetProperty("mode").GetString());
        Assert.Equal("greedy-e2e", w[3].GetProperty("mode").GetString());
        Assert.Equal("fixed-inference", w[4].GetProperty("mode").GetString());
        Assert.Equal(31, w[3].GetProperty("prefill_tokens").GetInt32());
        Assert.Equal("text", w[3].GetProperty("source").GetString());
        Assert.Equal(64, w[3].GetProperty("source_sha256").GetString().Length);
        Assert.Equal(JsonValueKind.Null, w[0].GetProperty("cold_verified").ValueKind);
        Assert.Equal(0, w[0].GetProperty("evictions").GetArrayLength());
        // The untimed correctness chain of a and b agree token for token; c has none.
        Assert.Equal(w[0].GetProperty("sampled_tokens").GetRawText(), w[1].GetProperty("sampled_tokens").GetRawText());
        Assert.Equal("untimed correctness", w[0].GetProperty("sampled_tokens").GetProperty("source").GetString());
        Assert.Equal(JsonValueKind.Null, w[2].GetProperty("sampled_tokens").ValueKind);
        Assert.Equal("run1", w[3].GetProperty("sampled_tokens").GetProperty("source").GetString());
        // Prefill-only runs have no decode rate (null, not 0).
        Assert.Equal(JsonValueKind.Null, w[5].GetProperty("samples")[0].GetProperty("decode_tps").ValueKind);
        foreach (JsonElement sample in w[0].GetProperty("samples").EnumerateArray())
        {
            Assert.True(sample.GetProperty("prefill_ms").GetDouble() >= 0);
            Assert.True(DateTime.Parse(sample.GetProperty("decode_start_utc").GetString()) >= DateTime.Parse(sample.GetProperty("prefill_start_utc").GetString()));
        }
    }

    [Fact]
    public void Prepare_RefusesAColdRow_ForAModelWithoutHostMappedTables_AndAShortOrForeignPrompt()
    {
        using var model = new SuiteModel(_gguf, tablePath: null);
        var cold = new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=c,prefill=8,cold", Defaults) };
        Assert.Contains("declares none", Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, cold)).Message);

        string tokens = Path.Combine(_dir, "t.json");
        File.WriteAllText(tokens, "[1, 2, 3]");
        var tooLong = new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=k,tokens=" + tokens + ",prefill=4", Defaults) };
        Assert.Contains("has only 3 tokens", Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, tooLong)).Message);
        File.WriteAllText(tokens, "[1, 2, 300]");
        var foreign = new List<BenchWorkloadSpec> { BenchWorkloadSpec.Parse("label=k,tokens=" + tokens, Defaults) };
        Assert.Contains("outside the vocabulary", Assert.Throws<ArgumentException>(() => TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, foreign)).Message);
    }

    /// <summary>A cold row on Linux: a warm row first caches the model's table (it reads it on every forward), then
    /// the cold row evicts it before its first run only (evict=row) or before every run (evict=run), and the JSON
    /// records the before/after residency. Other OSes refuse a cold row before the model loads (see Preflight).</summary>
    [Fact]
    public void ColdRows_EvictTheTable_BeforeTheirRuns_OnLinux()
    {
        if (!OperatingSystem.IsLinux())
        {
            Assert.False(HostPageCache.IsSupported);
            return;
        }
        string table = Path.Combine(_dir, "table.bin");
        var bytes = new byte[8L << 20];
        new Random(3).NextBytes(bytes);
        // Written through to disk before the model maps it: posix_fadvise(DONTNEED) skips dirty pages and pages under
        // writeback, so a table the kernel was still writing back (likelier while other tests write files) stayed
        // partly cached after the first eviction (4 of 8 MiB on WSL) and the row read as not verified.
        using (var stream = new FileStream(table, FileMode.CreateNew, FileAccess.Write))
        {
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }
        string json = Path.Combine(_dir, "cold.json");
        using var model = new SuiteModel(_gguf, table);
        var rows = new[] { "label=warm,prefill=8,decode=2,runs=1", "label=row,prefill=8,decode=2,runs=2,cold", "label=each,prefill=8,decode=2,runs=2,evict=run" }
            .Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        TensorSharp.Cli.Program.PreflightBenchmarkSuite(rows);
        TensorSharp.Cli.Program.RunBenchmarkSuite(model, TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, rows),
            new BenchSuiteOptions { JsonPath = json, EvictMeasure = HostPageCacheMeasure.Full });

        using var doc = JsonDocument.Parse(File.ReadAllText(json));
        var w = doc.RootElement.GetProperty("workloads").EnumerateArray().ToArray();
        Assert.Equal(0, w[0].GetProperty("evictions").GetArrayLength());
        Assert.Equal(1, w[1].GetProperty("evictions").GetArrayLength());
        Assert.Equal(2, w[2].GetProperty("evictions").GetArrayLength());
        foreach (JsonElement row in w.Skip(1))
        {
            Assert.True(row.GetProperty("cold_verified").GetBoolean());
            foreach (JsonElement e in row.GetProperty("evictions").EnumerateArray())
            {
                JsonElement node = Assert.Single(e.GetProperty("nodes").EnumerateArray());
                Assert.True(node.GetProperty("supported").GetBoolean());
                Assert.True(node.GetProperty("resident_before_mib").GetDouble() >= 7.9, e.GetRawText());
                Assert.Equal(0, node.GetProperty("resident_after_mib").GetDouble());
                Assert.True(node.GetProperty("mappings_advised").GetInt32() >= 1);
            }
        }
        Assert.Equal(new[] { 1, 2 }, w[2].GetProperty("evictions").EnumerateArray().Select(e => e.GetProperty("before_run").GetInt32()));
    }

    /// <summary>A cold row whose eviction cannot be verified (here: its table file is gone, so nothing can be
    /// measured; on Windows nothing can be evicted at all) is a warning by default, and stops the suite under
    /// --bench-evict-strict, with the JSON recording the failed row's eviction. The preflight that refuses cold rows
    /// on Windows is bypassed on purpose: this is the guard behind it.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnUnverifiedEviction_Warns_OrStopsTheSuiteWhenStrict(bool strict)
    {
        string table = Path.Combine(_dir, "gone.bin");
        File.WriteAllBytes(table, new byte[1 << 16]);
        string json = Path.Combine(_dir, $"strict-{strict}.json");
        using var model = new SuiteModel(_gguf, table);
        var rows = new[] { "label=cold,prefill=8,decode=1,runs=1,cold", "label=after,prefill=8,decode=1,runs=1" }
            .Select(s => BenchWorkloadSpec.Parse(s, Defaults)).ToList();
        var prepared = TensorSharp.Cli.Program.PrepareBenchmarkSuite(model, rows);
        model.DeleteTableBeforeNextEviction = true;
        var options = new BenchSuiteOptions { JsonPath = json, EvictMeasure = HostPageCacheMeasure.Full, EvictStrict = strict };
        if (strict)
            Assert.Contains("--bench-evict-strict", Assert.Throws<InvalidOperationException>(
                () => TensorSharp.Cli.Program.RunBenchmarkSuite(model, prepared, options)).Message);
        else
            TensorSharp.Cli.Program.RunBenchmarkSuite(model, prepared, options);

        using var doc = JsonDocument.Parse(File.ReadAllText(json));
        JsonElement root = doc.RootElement;
        if (strict)
        {
            Assert.Equal("failed", root.GetProperty("status").GetString());
            Assert.Equal(0, root.GetProperty("workloads").GetArrayLength());
            JsonElement failed = root.GetProperty("failed_workload");
            Assert.Equal("cold", failed.GetProperty("label").GetString());
            Assert.Equal(1, failed.GetProperty("evictions").GetArrayLength());
            Assert.NotEqual(JsonValueKind.True, failed.GetProperty("evictions")[0].GetProperty("verified").ValueKind);
        }
        else
        {
            Assert.Equal("completed", root.GetProperty("status").GetString());
            var w = root.GetProperty("workloads").EnumerateArray().ToArray();
            Assert.Equal(2, w.Length);
            Assert.NotEqual(JsonValueKind.True, w[0].GetProperty("cold_verified").ValueKind);
        }
    }

    // ---------------------------------------------------------------- the cross-node exchange

    /// <summary>The driver tells the worker to evict (control op 6, the measurement as payload), both evict, and the
    /// reports cross in one reduction that is also the barrier: the driver ends with node 0's and node 1's reports, and
    /// the next forward still reaches the worker in order.</summary>
    [Fact]
    public async Task Eviction_IsMirroredToTheWorker_AndBothReportsReachTheDriver()
    {
        var link = new PairLink();
        using var driver = new SuiteModel(_gguf, tablePath: null, new PairGroup(link, 0));
        using var worker = new SuiteModel(_gguf, tablePath: null, new PairGroup(link, 1));
        Task loop = Task.Run(worker.RunDistributedWorkerLoop);
        driver.BeginDistributedDriver();

        HostTableEvictionResult result = driver.EvictHostTablePageCache(HostPageCacheMeasure.Sampled);
        driver.Forward(new[] { 5, 6, 7 });
        link.Messages.Add((4, Array.Empty<int>()));
        await loop.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(2, result.NodeCount);
        Assert.Equal(new[] { 0, 1 }, result.Nodes.Select(n => n.Node));
        Assert.All(result.Nodes, n => Assert.Equal(HostPageCache.IsSupported, n.Supported));
        // No tables: nothing to evict, so nothing verifies it (Linux) or the OS cannot (elsewhere).
        Assert.NotEqual(true, result.Verified);
        Assert.Equal(new[] { 6, 1 }, link.Sent.Select(m => m.op));   // evict (payload: the measurement), then the forward
        Assert.Equal(new[] { (int)HostPageCacheMeasure.Sampled }, link.Sent[0].payload);
        Assert.Equal(new[] { 5, 6, 7 }, Assert.Single(worker.Prompts));
        Assert.Equal(1, link.Reductions);
    }

    [Fact]
    public void Eviction_StartedOffTheDriver_IsRefused_InsteadOfWaitingForever()
    {
        var link = new PairLink();
        using var notYetDriver = new SuiteModel(_gguf, tablePath: null, new PairGroup(link, 0));
        using var worker = new SuiteModel(_gguf, tablePath: null, new PairGroup(link, 1));
        Assert.Contains("Only the distributed driver", Assert.Throws<InvalidOperationException>(
            () => notYetDriver.EvictHostTablePageCache(HostPageCacheMeasure.None)).Message);
        Assert.Throws<InvalidOperationException>(() => worker.EvictHostTablePageCache(HostPageCacheMeasure.None));
        Assert.Empty(link.Sent);
        Assert.Equal(0, link.Reductions);
    }

    [Fact]
    public void EvictionSlots_RoundTrip_AndVerifiedNeedsEveryNode()
    {
        var a = new HostTableEvictionNode { Node = 1, Supported = true, Ranges = 1, BytesMiB = 128901.5, ResidentBeforeMiB = 61234.25,
            ResidentAfterMiB = 0, Verified = true, MappingsAdvised = 2, MappingsSkipped = 0, Errors = 0, ElapsedMs = 812.5, EvictMs = 300 };
        var slots = new float[HostTableEvictionNode.Fields * 2];
        a.WriteSlot(slots);
        Assert.Null(HostTableEvictionNode.ReadSlot(slots, 0));
        HostTableEvictionNode b = HostTableEvictionNode.ReadSlot(slots, 1);
        Assert.Equal(a.Describe(), b.Describe());

        HostTableEvictionNode Node(int k, bool? verified, bool supported = true) => new() { Node = k, Supported = supported, Verified = verified };
        Assert.True(new HostTableEvictionResult { NodeCount = 2, Nodes = new[] { Node(0, true), Node(1, true) } }.Verified);
        Assert.False(new HostTableEvictionResult { NodeCount = 2, Nodes = new[] { Node(0, true), Node(1, false) } }.Verified);
        Assert.False(new HostTableEvictionResult { NodeCount = 2, Nodes = new[] { Node(0, null), Node(1, null, supported: false) } }.Verified);
        Assert.Null(new HostTableEvictionResult { NodeCount = 2, Nodes = new[] { Node(0, true), Node(1, null) } }.Verified);
        Assert.Null(new HostTableEvictionResult { NodeCount = 2, Nodes = new[] { Node(0, true) } }.Verified);
    }

    /// <summary>A split checkpoint: a tensor's range names the shard that holds it (here the second, as the 1017's
    /// Engram table sits in its last shard) and the byte offset of its data in that shard, which is what a cold row
    /// evicts. Written by hand (no native quantizer), so it runs wherever the tests do.</summary>
    [Fact]
    public void GgufFile_NamesATensorsShardAndByteRange()
    {
        string first = Path.Combine(_dir, "split-00001-of-00002.gguf");
        string second = Path.Combine(_dir, "split-00002-of-00002.gguf");
        SyntheticGguf.Write(first, new List<SyntheticGguf.Kv>
        {
            new SyntheticGguf.Str { Key = "general.architecture", V = "probe" },
            new SyntheticGguf.U32 { Key = "split.count", V = 2 },
        }, new List<SyntheticGguf.Tensor> { SyntheticGguf.Gen("token_embd.weight", 0.1f, 16, 8) });
        SyntheticGguf.Write(second, new List<SyntheticGguf.Kv> { new SyntheticGguf.U32 { Key = "split.count", V = 2 } },
            new List<SyntheticGguf.Tensor>
            {
                SyntheticGguf.Gen("blk.5.ffn_norm.weight", 0.1f, 40),
                SyntheticGguf.Gen("per_layer_token_embd.weight", 0.1f, 64, 33),
            });

        using var file = new GgufFile(first);
        foreach ((string name, string shardPath) in new[] { ("per_layer_token_embd.weight", second), ("token_embd.weight", first) })
        {
            Assert.True(file.TryGetTensorFileRange(name, out string path, out long offset, out long bytes));
            Assert.Equal(Path.GetFullPath(shardPath), Path.GetFullPath(path));
            using var shard = GgufFile.OpenWithoutSiblingShards(shardPath);
            GgufTensorInfo info = shard.Tensors[name];
            Assert.Equal(shard.DataOffset + (long)info.Offset, offset);
            Assert.Equal(shard.GetTensorByteCount(info), bytes);
            Assert.True(offset + bytes <= new FileInfo(shardPath).Length);
        }
        Assert.True(file.TryGetTensorFileRange("per_layer_token_embd.weight", out _, out long tableOffset, out long tableBytes));
        Assert.Equal(64L * 33 * 4, tableBytes);
        Assert.True(tableOffset > 0);   // after blk.5.ffn_norm.weight in the second shard
        Assert.False(file.TryGetTensorFileRange("no.such.tensor", out _, out _, out _));
    }

    // ---------------------------------------------------------------- stand-ins

    /// <summary>Logits that depend on the whole sequence so far; records every prompt (a forward of more than one
    /// token). With a table path it maps that file and reads all of it on every forward, as a host-mapped lookup
    /// table is read, and declares it as its host-mapped table.</summary>
    private sealed unsafe class SuiteModel : ModelBase
    {
        private readonly List<int> _sequence = new();
        private MemoryMappedFile _map;
        private MemoryMappedViewAccessor _view;
        private readonly string _table;
        private readonly long _tableBytes;
        public ConcurrentQueue<int[]> Prompts { get; } = new();

        public SuiteModel(string path, string tablePath, ITensorParallelGroup group = null)
            : base(path, BackendType.Cpu, 1, group)
        {
            Config = new ModelConfig { Architecture = "suite-probe", VocabSize = Vocab, NumLayers = 1 };
            Tokenizer = new CharTokenizer();
            if (tablePath != null)
            {
                _table = tablePath;
                _tableBytes = new FileInfo(tablePath).Length;
                _map = MemoryMappedFile.CreateFromFile(tablePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
                _view = _map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            }
        }

        /// <summary>Delete the table file (unmapping it first) the next time the tables are listed: an eviction
        /// that then cannot measure anything.</summary>
        public bool DeleteTableBeforeNextEviction { get; set; }

        public override IReadOnlyList<HostFileRange> HostMappedTables
        {
            get
            {
                if (_table == null)
                    return Array.Empty<HostFileRange>();
                if (DeleteTableBeforeNextEviction)
                {
                    DeleteTableBeforeNextEviction = false;
                    _view?.Dispose();
                    _map?.Dispose();
                    _view = null;
                    _map = null;
                    File.Delete(_table);
                }
                return new[] { new HostFileRange("table", _table, 0, _tableBytes) };
            }
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (tokens.Length > 1)
                Prompts.Enqueue((int[])tokens.Clone());
            _sequence.AddRange(tokens);
            if (_view != null)
            {
                byte* p = null;
                _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
                try
                {
                    long n = _view.Capacity, s = 0;
                    for (long i = 0; i < n; i += 4096)
                        s += p[_view.PointerOffset + i];
                    GC.KeepAlive(s);
                }
                finally { _view.SafeMemoryMappedViewHandle.ReleasePointer(); }
            }
            var row = new float[Vocab];
            long h = 17;
            foreach (int t in _sequence) h = h * 31 + t;
            for (int i = 0; i < Vocab; i++)
                row[i] = (float)Math.Sin(0.37 * i + 0.001 * (h % 100003));
            return row;
        }

        protected override void ResetKVCacheCore() => _sequence.Clear();

        public override void Dispose()
        {
            _view?.Dispose();
            _map?.Dispose();
            base.Dispose();
        }
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, BenchmarkSuiteTests.Vocab).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => BenchmarkSuiteTests.Vocab;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }

    /// <summary>A driver-to-worker control channel and a two-node sum reduction, in memory.</summary>
    private sealed class PairLink
    {
        public BlockingCollection<(int op, int[] payload)> Messages { get; } = new();
        public List<(int op, int[] payload)> Sent { get; } = new();
        public int Reductions;
        private readonly Barrier _barrier = new(2);
        private readonly float[][] _parts = new float[2][];

        public void Reduce(int node, float[] buffer, int count)
        {
            _parts[node] = buffer[..count];
            _barrier.SignalAndWait(TimeSpan.FromSeconds(30));
            for (int i = 0; i < count; i++)
                buffer[i] = _parts[0][i] + _parts[1][i];
            if (node == 0) Interlocked.Increment(ref Reductions);
            _barrier.SignalAndWait(TimeSpan.FromSeconds(30));
        }
    }

    private sealed class PairGroup(PairLink link, int node) : ITensorParallelGroup, INestedTensorParallelGroup
    {
        public ITensorParallelGroup LocalGroup => this;
        public int Degree => 1;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => node;
        public int NodeCount => 2;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void CrossNodeAllReduce(float[] buffer, int count) => link.Reduce(node, buffer, count);
        public void Synchronize() { }
        public void Barrier() => throw new NotSupportedException("the exchange must use the reduction");
        public void BroadcastControl(int op, int[] payload)
        {
            lock (link.Sent) link.Sent.Add((op, (int[])payload.Clone()));
            link.Messages.Add((op, (int[])payload.Clone()));
        }
        public (int op, int[] payload) ReceiveControl() => link.Messages.Take();
        public void Dispose() { }
    }
}
