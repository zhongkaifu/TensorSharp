// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A cold --bench-workload row in a distributed run: the driver drops the Engram tables from every node's page cache
// between forwards (ModelBase.EvictHostTablePageCache: a control op to the workers, then one cross-node reduction that
// carries every node's report). On the two-node loopback over real TCP it must reach the worker in order with the
// forwards around it, and change no logit. Before it drops anything, each node stops the reads its engine makes of the
// tables on its own (the read-ahead, a background warm), so nothing reads the dropped pages straight back.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed partial class Dsv41CudaDistributedTests
{
    [CudaFact]
    public async Task EngramEviction_BetweenForwards_ReachesEveryNode_AndChangesNoLogit()
    {
        float[][] reference = Reference();
        var results = new List<HostTableEvictionResult>();
        IReadOnlyList<HostFileRange> tables = null;
        string fixture = Dsv41SyntheticModelBuilder.Fixture();
        Run run = await Distributed(3, driver => Task.Run(() =>
        {
            tables = driver.HostMappedTables;
            int calls = 0;
            float[] Forward(int[] tokens)
            {
                // Before the prefill, then before every sixth call (decode steps and the append included).
                if (calls++ % 6 == 0)
                    results.Add(driver.EvictHostTablePageCache(HostPageCacheMeasure.Full));
                return driver.Forward(tokens);
            }
            void Reset()
            {
                results.Add(driver.EvictHostTablePageCache(HostPageCacheMeasure.Sampled));
                driver.ResetKVCache();
            }
            return Script(Forward, Reset);
        }));

        AssertBitIdentical("Engram evictions between forwards", reference, run.Logits);
        Assert.True(results.Count >= 5, $"{results.Count} evictions");
        foreach (HostTableEvictionResult r in results)
        {
            Assert.Equal(2, r.NodeCount);
            Assert.Equal(new[] { 0, 1 }, r.Nodes.Select(n => n.Node));
            Assert.All(r.Nodes, n => Assert.Equal(HostPageCache.IsSupported, n.Supported));
            if (HostPageCache.IsSupported)
                Assert.True(r.Verified, string.Join("; ", r.Nodes.Select(n => n.Describe())));
            else
                Assert.False(r.Verified);
        }

        // The table a cold row evicts is the checkpoint's Engram table, named the same on every node.
        HostFileRange table = Assert.Single(tables);
        Assert.Equal("per_layer_token_embd.weight", table.Name);
        using var file = new GgufFile(fixture);
        Assert.True(file.TryGetTensorFileRange(table.Name, out string path, out long offset, out long bytes));
        Assert.Equal((path, offset, bytes), (table.Path, table.Offset, table.Bytes));
    }

    /// <summary>
    /// An eviction first stops what the direct engine reads from the tables on its own (ModelBase.QuiesceHostTableReaders):
    /// the last forward's read-ahead, slowed here to about 20 ms a row so it is still running when the eviction starts,
    /// is abandoned and waited for, and nothing more is read once the eviction returns, so a cold row is not warmed
    /// back behind the benchmark's back. The script's logits stay those of the single-process engine. (The quiesce runs
    /// on every OS; only the page-cache drop itself is Linux-only.)
    /// </summary>
    [CudaFact]
    public void AnEviction_StopsTheReadAhead_BeforeItDropsTheTables()
    {
        float[][] reference = Reference();
        _env.Set("TS_DSV41_ENGRAM_PREFETCH", "1");
        _env.Set("TS_DSV41_ENGRAM_WARM", "0");
        using ModelBase model = ModelBase.Create(Dsv41SyntheticModelBuilder.Fixture(), BackendType.Cuda);
        Dsv41EngramReader reader = ((DeepSeek4Model)model).CudaExecutor.EngramReader;
        Assert.True(reader.Prefetch);
        reader.ReadHookForTests = kind =>
        {
            if (kind == 0)
                Thread.Sleep(20);
        };
        var rows = new float[Calls.Length][];
        rows[0] = (float[])model.Forward(Calls[0].Tokens).Clone();
        Assert.True(reader.ReadingAheadThreads > 0, "the prefill's read-ahead had already finished; the test proves nothing");
        long before = reader.RowsTouched;
        HostTableEvictionResult result = model.EvictHostTablePageCache(HostPageCacheMeasure.None);
        Assert.Equal(0, reader.ReadingAheadThreads);
        long touched = reader.RowsTouched;
        int ahead = Calls[0].Tokens.Length * (Dsv41SyntheticModelBuilder.EngramNgram - 1) * Dsv41SyntheticModelBuilder.EngramHeads
            * Dsv41SyntheticModelBuilder.EngramLayers.Length;
        Assert.True(touched < ahead, $"{touched} of {ahead} rows were read ahead: the eviction did not cut the read-ahead short");
        Thread.Sleep(200);
        Assert.Equal(touched, reader.RowsTouched);
        Assert.Equal(HostPageCache.IsSupported, Assert.Single(result.Nodes).Supported);
        _output.WriteLine($"read ahead {before} rows by the end of the prefill, {touched} of {ahead} when the eviction returned");

        reader.ReadHookForTests = null;
        for (int i = 1; i < Calls.Length; i++)
        {
            if (Calls[i].Reset)
                model.ResetKVCache();
            rows[i] = (float[])model.Forward(Calls[i].Tokens).Clone();
        }
        AssertBitIdentical("an eviction during the read-ahead", reference, rows);
    }

    /// <summary>
    /// A reset abandons the last forward's read-ahead (slowed to about 40 ms a row here): its threads leave after the row
    /// they are on, long before the read-ahead would have finished, and the script's logits stay those of the
    /// single-process engine.
    /// </summary>
    [CudaFact]
    public void AReset_StopsTheReadAhead()
    {
        float[][] reference = Reference();
        _env.Set("TS_DSV41_ENGRAM_PREFETCH", "1");
        _env.Set("TS_DSV41_ENGRAM_WARM", "0");
        using ModelBase model = ModelBase.Create(Dsv41SyntheticModelBuilder.Fixture(), BackendType.Cuda);
        Dsv41EngramReader reader = ((DeepSeek4Model)model).CudaExecutor.EngramReader;
        reader.ReadHookForTests = kind =>
        {
            if (kind == 0)
                Thread.Sleep(40);
        };
        var rows = new List<float[]>();
        int i = 0;
        for (; !Calls[i].Reset; i++)
            rows.Add((float[])model.Forward(Calls[i].Tokens).Clone());
        // The last forward before the reset was a one-token decode: give the read-ahead a long forward to cut short.
        model.ResetKVCache();
        rows.Add((float[])model.Forward(Calls[i].Tokens).Clone());
        Assert.True(reader.ReadingAheadThreads > 0, "the prompt's read-ahead had already finished; the test proves nothing");
        model.ResetKVCache();
        Assert.True(SpinWait.SpinUntil(() => reader.ReadingAheadThreads == 0, TimeSpan.FromSeconds(1)),
            "the read-ahead kept going after the reset");
        int ahead = Calls[i].Tokens.Length * (Dsv41SyntheticModelBuilder.EngramNgram - 1) * Dsv41SyntheticModelBuilder.EngramHeads
            * Dsv41SyntheticModelBuilder.EngramLayers.Length;
        long touched = reader.RowsTouched;
        Assert.True(touched < ahead * 2L, $"{touched} rows read ahead: the reset did not cut the read-ahead short");
        reader.ReadHookForTests = null;
        // The reset above cleared the sequence: run the re-prefill again, then the rest of the script.
        rows[^1] = (float[])model.Forward(Calls[i].Tokens).Clone();
        for (i++; i < Calls.Length; i++)
            rows.Add((float[])model.Forward(Calls[i].Tokens).Clone());
        AssertBitIdentical("a reset during the read-ahead", reference, rows.ToArray());
    }
}
