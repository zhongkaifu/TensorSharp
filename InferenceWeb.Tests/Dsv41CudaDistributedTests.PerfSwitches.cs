// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Every DeepSeek V4.1 direct-CUDA speed switch of S10/S11 together, in one build. Each package proved its own switch on
// and off; here each switch is turned on alone, then all of them at once, then nothing is set (the shipped defaults),
// and every run's logits must be byte-identical to the S9 behaviour (every switch off). The script is PipelineSteps:
// whole prompts of 1 to 5 ubatches (odd and even counts, exact multiples and remainders), a prompt fed in three chunks,
// decode steps after each prefill, and resets. Three topologies:
//  - one process, host-mapped Engram tables (TS_DSV41_ENGRAM_DEVICE=0, set by the class): the Engram reader's threads,
//    prefetch and random-access advice, the warm, and dropping weight pages during the load;
//  - two nodes of one GPU over TCP on 127.0.0.1, cut after layer 4 (both Engram layers on the worker, the 1017's shape):
//    the same switches plus the table owner (TS_DSV41_ENGRAM_OWNER), the deferred send (TS_DSV4_ASYNC_SEND 1 and 2) and
//    a cold-row page-cache eviction on every node before every forward (ModelBase.EvictHostTablePageCache);
//  - two nodes of two GPUs in process (the 1017's 2 x 2 path on the span fixture), every switch but the eviction.
// Each run also checks that the switch took effect (the reader's settings and the rows it read ahead, the owners and the
// Engram messages, the deferred payloads, the load's drop decision, the warm), so "on" cannot quietly be "off". The
// shipped defaults turn on only what is bit-identical and low-risk (the parallel reads and MADV_RANDOM; the warm and
// the load's drop rule keep the native loader's automatic rules); the prefetch, the balanced owner and the deferred
// send stay off until real hardware has measured them, so "nothing set" must show them off. On
// Windows the Linux-only parts (posix_madvise, posix_fadvise, mincore, the eviction itself) do nothing: there these runs
// prove the plumbing and the message order, not the kernel calls.
using System.Reflection;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed partial class Dsv41CudaDistributedTests
{
    private const string EngramThreadsVariable = "TS_DSV41_ENGRAM_THREADS", EngramPrefetchVariable = "TS_DSV41_ENGRAM_PREFETCH",
        EngramRandomVariable = "TS_DSV41_ENGRAM_RANDOM", EngramOwnerVariable = "TS_DSV41_ENGRAM_OWNER",
        EngramWarmVariable = "TS_DSV41_ENGRAM_WARM", EngramWarmThreadsVariable = "TS_DSV41_ENGRAM_WARM_THREADS",
        LoadDropCacheVariable = "TS_DSV4_LOAD_DROP_CACHE";

    /// <summary>The S9 behaviour: every speed switch off.</summary>
    private static readonly IReadOnlyDictionary<string, string> EverySwitchOff = new Dictionary<string, string>
    {
        [EngramThreadsVariable] = "1", [EngramPrefetchVariable] = "0", [EngramRandomVariable] = "0",
        [EngramOwnerVariable] = "local", [EngramWarmVariable] = "0", [EngramWarmThreadsVariable] = null,
        [LoadDropCacheVariable] = "0", [AsyncSendVariable] = "0",
    };

    private static readonly (string Name, string Value)[] EverySwitchOn =
    {
        (EngramThreadsVariable, "16"), (EngramPrefetchVariable, "1"), (EngramRandomVariable, "1"),
        (EngramOwnerVariable, "balance"), (EngramWarmVariable, "1"), (EngramWarmThreadsVariable, "4"),
        (LoadDropCacheVariable, "1"), (AsyncSendVariable, "1"),
    };

    /// <summary>One setting: the variables on top of every switch off (null: unset), and whether the driver evicts the
    /// Engram tables from every node's page cache before every forward.</summary>
    private sealed record PerfSetting(string Name, IReadOnlyDictionary<string, string> Env, bool Evict = false)
    {
        public string Value(string name) => Env.TryGetValue(name, out string v) ? v : null;
    }

    private static PerfSetting SwitchesOn(string name, bool evict, params (string Name, string Value)[] on)
    {
        var env = new Dictionary<string, string>(EverySwitchOff);
        foreach (var (n, v) in on)
            env[n] = v;
        return new PerfSetting(name, env, evict);
    }

    /// <summary>Every switch off, each switch on alone, every switch on, and nothing set. The owner and the deferred send
    /// only exist on a distributed stage; the eviction only through a model's driver.</summary>
    private static PerfSetting[] PerfSettings(bool distributed, bool evict)
    {
        var settings = new List<PerfSetting>
        {
            SwitchesOn("every switch off (S9)", false),
            SwitchesOn("Engram threads 16", false, (EngramThreadsVariable, "16")),
            SwitchesOn("Engram prefetch", false, (EngramPrefetchVariable, "1")),
            SwitchesOn("Engram random-access advice", false, (EngramRandomVariable, "1")),
            SwitchesOn("Engram warm (synchronous, 4 threads)", false, (EngramWarmVariable, "1"), (EngramWarmThreadsVariable, "4")),
            SwitchesOn("drop weight pages during the load", false, (LoadDropCacheVariable, "1")),
        };
        if (distributed)
        {
            settings.Add(SwitchesOn("Engram owner balance", false, (EngramOwnerVariable, "balance")));
            settings.Add(SwitchesOn("deferred send at the next hand-over (1)", false, (AsyncSendVariable, "1")));
            settings.Add(SwitchesOn("deferred send at the next send (2)", false, (AsyncSendVariable, "2")));
        }
        if (evict)
            settings.Add(SwitchesOn("page-cache eviction before every forward", true));
        settings.Add(SwitchesOn("every switch on", evict, EverySwitchOn));
        if (distributed)
            settings.Add(SwitchesOn("every switch on, deferred send 2", evict,
                EverySwitchOn.Select(p => p.Name == AsyncSendVariable ? (p.Name, "2") : p).ToArray()));
        settings.Add(new PerfSetting("the defaults (nothing set)", EverySwitchOff.ToDictionary(p => p.Key, _ => (string)null), evict));
        return settings.ToArray();
    }

    private void Apply(PerfSetting setting)
    {
        foreach (var (name, value) in setting.Env)
            _env.Set(name, value);
    }

    /// <summary>What one executor did with the switches. The prefetch reads ahead only for the layers a stage runs, from
    /// the tables it keeps: a stage that keeps a table for another node's layer reads its rows when it sends them.</summary>
    private sealed record PerfFacts(int Threads, bool Prefetch, bool RandomAdvice, long RowsReadAhead, bool DropsPages,
        TaskStatus? Warm, bool MapsAnyTable, bool RunsEngramLayers)
    {
        public override string ToString() =>
            $"threads {Threads}, prefetch {Prefetch} ({RowsReadAhead} rows read ahead), advice {RandomAdvice}, drops weight pages " +
            $"{DropsPages}, warm {(Warm?.ToString() ?? "none")}, maps a table {MapsAnyTable}, runs Engram layers {RunsEngramLayers}";
    }

    private static PerfFacts FactsOf(DeepSeek4CudaExecutor executor, int tables)
    {
        Dsv41EngramReader reader = executor.EngramReader;
        const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;
        var shards = (GgufShardSet)typeof(DeepSeek4CudaExecutor).GetField("_gguf", Private)!.GetValue(executor);
        var warm = (Task)typeof(DeepSeek4CudaExecutor).GetField("_engramWarm", Private)!.GetValue(executor);
        return new PerfFacts(reader.Threads, reader.Prefetch, reader.RandomAdvice, reader.RowsTouched, shards.DropPagesAfterRead,
            warm?.Status, Enumerable.Range(0, tables).Any(executor.MapsEngramTable), executor.Engine.EngramRuns.Any(r => r > 0));
    }

    /// <summary>The facts every executor of a run must show for <paramref name="setting"/>.</summary>
    private static void AssertTookEffect(PerfSetting setting, PerfFacts facts, string where)
    {
        string threads = setting.Value(EngramThreadsVariable);
        int expectedThreads = threads != null ? int.Parse(threads) : Math.Min(16, Environment.ProcessorCount);
        Assert.True(expectedThreads == facts.Threads, $"[{setting.Name}] {where}: {facts}");
        // Unset: the prefetch is off; the advice is on with more than one thread.
        bool Switch(string name, bool unset) => setting.Value(name) is { } v ? v == "1" : unset;
        Assert.True(Switch(EngramPrefetchVariable, false) == facts.Prefetch, $"[{setting.Name}] {where}: prefetch: {facts}");
        Assert.True(Switch(EngramRandomVariable, facts.Threads > 1) == facts.RandomAdvice, $"[{setting.Name}] {where}: advice: {facts}");
        if (facts.Prefetch && facts.MapsAnyTable && facts.RunsEngramLayers)
            Assert.True(facts.RowsReadAhead > 0, $"[{setting.Name}] {where}: the prefetch read nothing ahead: {facts}");
        if (!facts.Prefetch)
            Assert.True(facts.RowsReadAhead == 0, $"[{setting.Name}] {where}: rows read ahead with the prefetch off: {facts}");
        if (setting.Value(LoadDropCacheVariable) is { } drop)
            Assert.True((drop == "1") == facts.DropsPages, $"[{setting.Name}] {where}: drop decision: {facts}");
        switch (setting.Value(EngramWarmVariable))
        {
            case "0":
                Assert.True(facts.Warm == null, $"[{setting.Name}] {where}: warmed with the warm off: {facts}");
                break;
            case "1" when facts.MapsAnyTable:
                Assert.True(facts.Warm == TaskStatus.RanToCompletion, $"[{setting.Name}] {where}: synchronous warm: {facts}");
                break;
        }
    }

    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, float[][]> SwitchesOffReferences = new();

    /// <summary>The script on the single-process engine with every switch off (the S9 behaviour).</summary>
    private float[][] SwitchesOffReference(string path) => SwitchesOffReferences.GetOrAdd(path, p =>
    {
        Apply(PerfSettings(distributed: false, evict: false)[0]);
        using ModelBase model = ModelBase.Create(p, BackendType.Cuda);
        return RunPipelineForwards(model.Forward, model.ResetKVCache);
    });

    private static float[][] RunPipelineForwards(Func<int[], float[]> forward, Action reset, Action beforeForward = null)
    {
        var rows = new List<float[]>();
        foreach (Step step in PipelineForwards)
        {
            if (step.Reset)
                reset();
            beforeForward?.Invoke();
            rows.Add((float[])forward(step.Tokens).Clone());
        }
        return rows.ToArray();
    }

    /// <summary>One process, host-mapped tables: each Engram switch alone, all of them, nothing set.</summary>
    [CudaFact]
    public void PerfSwitches_OnOneProcess_AnswerAsEverySwitchOff_ToTheBit()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        int tables = Dsv41SyntheticModelBuilder.EngramLayers.Length;
        float[][] off = null;
        foreach (PerfSetting setting in PerfSettings(distributed: false, evict: false))
        {
            Apply(setting);
            using ModelBase model = ModelBase.Create(path, BackendType.Cuda);
            DeepSeek4CudaExecutor executor = ((DeepSeek4Model)model).CudaExecutor;
            Assert.False(executor.Engine.EngramResident, "TS_DSV41_ENGRAM_DEVICE=0 left the tables in VRAM");
            float[][] logits = RunPipelineForwards(model.Forward, model.ResetKVCache);
            PerfFacts facts = FactsOf(executor, tables);
            AssertTookEffect(setting, facts, "one process");
            if (off == null)
                off = logits;
            else
                AssertSameBytes($"[one process] {setting.Name} vs every switch off", off, logits);
            _output.WriteLine($"[one process] {setting.Name}: byte-identical; {facts}");
        }
    }

    /// <summary>Two nodes of one GPU over TCP, both Engram layers on the worker: every switch alone (the eviction included),
    /// every switch on, nothing set; and the single-process engine with every switch off.</summary>
    [CudaFact]
    public async Task PerfSwitches_OnTwoNodesOverTcp_AnswerAsEverySwitchOff_ToTheBit()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        int tables = Dsv41SyntheticModelBuilder.EngramLayers.Length;
        float[][] reference = SwitchesOffReference(path);
        int forwards = PipelineForwards.Count() + 1;   // + the warm-up's one-token forward
        float[][] off = null;
        foreach (PerfSetting setting in PerfSettings(distributed: true, evict: true))
        {
            Apply(setting);
            var evictions = new List<HostTableEvictionResult>();
            PerfFacts driverFacts = null;
            Run run = await Distributed(new[] { 0, 5, Dsv41SyntheticModelBuilder.Layers }, path, driver => Task.Run(() =>
            {
                float[][] rows = RunPipelineForwards(driver.Forward, driver.ResetKVCache, setting.Evict
                    ? () => evictions.Add(driver.EvictHostTablePageCache(HostPageCacheMeasure.Sampled))
                    : null);
                driverFacts = FactsOf(driver.CudaExecutor, tables);
                return rows;
            }));
            AssertSameBytes($"[2 nodes, TCP] {setting.Name} vs the single-process engine with every switch off", reference, run.Logits);
            if (off == null)
                off = run.Logits;
            else
                AssertSameBytes($"[2 nodes, TCP] {setting.Name} vs every switch off", off, run.Logits);

            AssertTookEffect(setting, driverFacts, "driver");
            StageCounters worker = run.Worker, driverStage = run.Driver;
            // Unset: the tables stay with their reader, and the sends are synchronous.
            bool balance = setting.Value(EngramOwnerVariable) == "balance";
            bool defer = setting.Value(AsyncSendVariable) is "1" or "2";
            Assert.Equal(balance ? new[] { 0, 1 } : new[] { 0, 0 }, worker.EngramOwners);
            Assert.Equal(balance ? new[] { true, false } : new[] { true, true }, worker.MapsEngram);
            Assert.Equal(balance ? new[] { false, true } : new[] { false, false }, driverStage.MapsEngram);
            Assert.Equal(balance ? forwards : 0, driverStage.EngramSent);
            Assert.Equal(balance ? forwards : 0, worker.EngramReceived);
            Assert.Equal(defer ? ExpectedDeferred : 0, worker.Deferred);
            Assert.Equal(0, driverStage.Deferred);
            if (setting.Evict)
            {
                Assert.Equal(PipelineForwards.Count(), evictions.Count);
                Assert.All(evictions, e =>
                {
                    Assert.Equal(2, e.NodeCount);
                    Assert.Equal(new[] { 0, 1 }, e.Nodes.Select(n => n.Node));
                    Assert.All(e.Nodes, n => Assert.Equal(HostPageCache.IsSupported, n.Supported));
                    // Both nodes live in this process, so their evictions take turns (HostPageCache serializes them);
                    // on Linux every node's report must be clean and verified, on any other OS none can be.
                    if (HostPageCache.IsSupported)
                        Assert.All(e.Nodes, n => Assert.True(n.Errors == 0 && n.MappingsAdvised >= 1 && n.Verified == true,
                            $"node {n.Node}: {n.Errors} errors, {n.MappingsAdvised} mappings advised, verified {n.Verified}, " +
                            $"{n.ResidentAfterMiB * 1024:F1} KiB left"));
                    else
                        Assert.False(e.Verified);
                });
            }
            _output.WriteLine($"[2 nodes, TCP] {setting.Name}: byte-identical to every switch off and to one process; driver: " +
                $"{driverFacts}; owners [{string.Join(",", worker.EngramOwners)}], Engram messages {driverStage.EngramSent}, " +
                $"deferred sends {worker.Deferred}, evictions {evictions.Count}" +
                (evictions.Count > 0 ? $" (supported {evictions[0].Nodes[0].Supported}, node reports verified " +
                    $"{evictions.Sum(e => e.Nodes.Count(n => n.Verified == true))} of {evictions.Sum(e => e.Nodes.Count)}, most left " +
                    $"{evictions.SelectMany(e => e.Nodes).Max(n => n.ResidentAfterMiB) * 1024:F0} KiB)" : ""));
        }
    }

    /// <summary>Two nodes of two GPUs in process on the span fixture (the 1017's 2 x 2 path, both Engram layers on stage
    /// 0): every switch alone, every switch on, nothing set; and the single-process engine with every switch off.</summary>
    [CudaFact]
    public async Task PerfSwitches_OnTwoNodesOfTwoGpus_AnswerAsEverySwitchOff_ToTheBit()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture(Span);
        int tables = Span.EngramLayers.Length;
        float[][] reference = SwitchesOffReference(path);
        int forwards = PipelineForwards.Count();
        float[][] off = null;
        foreach (PerfSetting setting in PerfSettings(distributed: true, evict: false))
        {
            Apply(setting);
            var facts = new PerfFacts[2];
            LocalRun run = await LocalStages(path, new[] { 0, 3, 5, 7, 8 }, localDevices: 2, drive: (executors, forward) =>
            {
                var rows = new List<float[]>();
                foreach (Step step in PipelineSteps)
                {
                    if (step.Reset)
                        foreach (var ex in executors)
                            ex.Reset();
                    if (step.Tokens == null)
                    {
                        foreach (var ex in executors)
                            Assert.False(ex.Truncate(step.TruncateTo), $"a stage truncated to {step.TruncateTo}");
                        continue;
                    }
                    rows.Add((float[])forward(step.Tokens).Clone());
                }
                DeepSeek4CudaExecutor[] byStage = executors.OrderBy(e => e.Engine.OwnedLayers.First).ToArray();
                for (int s = 0; s < 2; s++)
                    facts[s] = FactsOf(byStage[s], tables);
                return rows.ToArray();
            });
            AssertSameBytes($"[2 x 2] {setting.Name} vs the single-process engine with every switch off", reference, run.Logits);
            if (off == null)
                off = run.Logits;
            else
                AssertSameBytes($"[2 x 2] {setting.Name} vs every switch off", off, run.Logits);

            for (int s = 0; s < 2; s++)
                AssertTookEffect(setting, facts[s], $"stage {s}");
            StageCounters first = run.Stages[0], last = run.Stages[1];
            // Unset: the tables stay with their reader, and the sends are synchronous.
            bool balance = setting.Value(EngramOwnerVariable) == "balance";
            bool defer = setting.Value(AsyncSendVariable) is "1" or "2";
            Assert.Equal(balance ? new[] { 0, 1 } : new[] { 0, 0 }, first.EngramOwners);
            Assert.Equal(balance ? new[] { true, false } : new[] { true, true }, first.MapsEngram);
            Assert.Equal(balance ? new[] { false, true } : new[] { false, false }, last.MapsEngram);
            Assert.Equal(balance ? forwards : 0, last.EngramSent);
            Assert.Equal(balance ? forwards : 0, first.EngramReceived);
            Assert.Equal(defer ? ExpectedDeferred : 0, first.Deferred);
            Assert.Equal(0, last.Deferred);
            _output.WriteLine($"[2 x 2] {setting.Name}: byte-identical to every switch off and to one process; stage 0: {facts[0]}; " +
                $"stage 1: {facts[1]}; owners [{string.Join(",", first.EngramOwners)}], Engram messages {last.EngramSent}, " +
                $"deferred sends {first.Deferred}");
        }
    }
}
