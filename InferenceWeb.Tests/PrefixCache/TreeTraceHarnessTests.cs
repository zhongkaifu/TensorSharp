// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// The tree-level property harness (DESIGN §12.3, M1). Seeds run in parallel; each seed is an
/// independent 300-operation trace against its own tree.
/// <list type="bullet">
/// <item><c>PREFIX_CACHE_TREE_SEEDS=&lt;n&gt;</c> sets how many seeds run (default 1,000 per PR run;
/// the M1 exit criterion and the nightly run use <c>PREFIX_CACHE_TREE_SEEDS=20000</c>).</item>
/// <item><c>PREFIX_CACHE_TREE_SEED_START=&lt;n&gt;</c> offsets the seed range.</item>
/// <item><c>PREFIX_CACHE_SEED=&lt;n&gt;</c> reruns exactly one seed (replay a failure).</item>
/// </list>
/// These are test-harness variables only; the product reads none of them.
/// </summary>
[Trait("Category", "PrefixCacheProperty")]
public class TreeTraceHarnessTests
{
    internal const int DefaultSeeds = 1000;
    private readonly ITestOutputHelper _output;

    public TreeTraceHarnessTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void TreeTraceHarness_AllSeedsClean()
    {
        int start = ReadInt("PREFIX_CACHE_TREE_SEED_START", 0);
        int count = ReadInt("PREFIX_CACHE_TREE_SEEDS", DefaultSeeds);
        string? single = Environment.GetEnvironmentVariable("PREFIX_CACHE_SEED");
        int[] seeds = int.TryParse(single, out int one) ? new[] { one } : Enumerable.Range(start, count).ToArray();

        var failures = new ConcurrentBag<(int Seed, string Message)>();
        long plans = 0, comparisons = 0, acquires = 0, finishes = 0;
        var sw = Stopwatch.StartNew();
        Parallel.ForEach(seeds, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, seed =>
        {
            var h = new TreeTraceHarness(seed);
            try
            {
                h.Run();
            }
            catch (TreeTraceHarness.HarnessFailure ex)
            {
                failures.Add((seed, ex.Message));
            }
            Interlocked.Add(ref plans, h.Plans);
            Interlocked.Add(ref comparisons, h.Comparisons);
            Interlocked.Add(ref acquires, h.Acquires);
            Interlocked.Add(ref finishes, h.Finishes);
        });
        _output.WriteLine($"{seeds.Length} seeds in {sw.Elapsed.TotalSeconds:F1} s: {plans} plans, {comparisons} brute-force comparisons, {acquires} acquires, {finishes} finishes, {failures.Count} failing seeds");
        if (!failures.IsEmpty)
        {
            (int seed, string message) = failures.OrderBy(f => f.Seed).First();
            Assert.Fail($"{failures.Count} of {seeds.Length} seeds failed (replay with PREFIX_CACHE_SEED={seed}). First failure:\n{message}");
        }
        Assert.True(comparisons > seeds.Length, "the harness compared almost no plans");
    }

    private static int ReadInt(string name, int fallback) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;
}
