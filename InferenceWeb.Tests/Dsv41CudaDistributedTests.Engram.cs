// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Per-node Engram ownership (S10 package 2): under a cross-node layer split each host-mapped Engram table is kept by
// one node (Dsv4StagePlacement.EngramOwners), which maps it, keeps it warm and gathers its rows; a table whose layer
// runs on another node crosses once per forward as raw rows, before any boundary, and the reading node dequantizes
// them with the same host function as its own gather. With TS_DSV41_ENGRAM_OWNER=balance on the 1017 at (12, 22, 31)
// both tables are read on node 1, so node 0 keeps layer 14's 63 GiB slice and each node holds one. Unset is 'local'
// (the S9 placement): balance is opt-in until a two-node run on real hardware has measured it.
using System.Net;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Distributed;

namespace InferenceWeb.Tests;

public sealed partial class Dsv41CudaDistributedTests
{
    // ================================================================ owners (no GPU)

    /// <summary>
    /// From the 1017's header bytes on 2 nodes x 2 GPUs (driver last): both tables are read on stage 0 (node 1, layers
    /// 0-21). TS_DSV41_ENGRAM_OWNER=balance keeps table 0 (layer 1) there and hands table 1 (layer 14) to stage 1
    /// (node 0), so each node holds one 63 GiB slice; unset, empty or 'local' keeps both with their reader (the S9
    /// placement, the default); any other value is refused.
    /// </summary>
    [Fact]
    public void The1017Tables_AreKeptOnePerNode()
    {
        Layout1017 l = Load1017();
        int[] device = Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 2, 2, driverRunsLastStage: true);
        Assert.Equal(new[] { 1, 14 }, Enumerable.Range(0, l.Topology.NLayer).Where(il => l.Topology.EngramIndex[il] >= 0));
        long[] slices = { 67_584_000_000, 67_587_000_000 };   // 62.94 / 62.95 GiB
        Assert.Equal(new[] { 0, 1 }, Dsv4StagePlacement.EngramOwners(l.Topology, device, 2, 2, slices, "balance"));
        Assert.Equal(new[] { 0, 1 }, Dsv4StagePlacement.EngramOwners(l.Topology, device, 2, 2, slices, " balance "));
        foreach (string local in new[] { null, "", " ", "local" })
            Assert.Equal(new[] { 0, 0 }, Dsv4StagePlacement.EngramOwners(l.Topology, device, 2, 2, slices, local));
        foreach (string bad in new[] { "node0", "Balance", "auto" })
            Assert.Contains("TS_DSV41_ENGRAM_OWNER", Assert.Throws<NotSupportedException>(() =>
                Dsv4StagePlacement.EngramOwners(l.Topology, device, 2, 2, slices, bad)).Message);
    }

    /// <summary>
    /// The startup digest names the owner mode by meaning: a node that leaves TS_DSV41_ENGRAM_OWNER unset and one that
    /// says 'local' (or ' local ') agree; 'balance' differs from both; an unknown value stays as given, so it can never
    /// agree with a valid one by accident (and EngramOwners refuses it).
    /// </summary>
    [Fact]
    public void TheEngramOwnerDigest_ReadsUnsetAsLocal()
    {
        static string Digest(string owner) => DeepSeek41Architecture.DirectCudaNumericalOptions(
            name => name == "TS_DSV41_ENGRAM_OWNER" ? owner : null);
        string unset = Digest(null);
        Assert.Contains("|TS_DSV41_ENGRAM_OWNER=local", unset);
        Assert.Equal(unset, Digest("local"));
        Assert.Equal(unset, Digest(" local "));
        Assert.Equal(unset, Digest(""));
        Assert.NotEqual(unset, Digest("balance"));
        Assert.Equal(Digest("balance"), Digest(" balance"));
        Assert.Contains("|TS_DSV41_ENGRAM_OWNER=node0", Digest("node0"));
    }

    /// <summary>
    /// On the fixture (Engram on layers 1 and 3), the balanced ownership moves a table only where both are read on one
    /// stage: at the cut after layer 4 (both on the worker) table 1 goes to the driver; at the cuts after layers 1 and 2
    /// each table already sits with its own reader. A tie stays with the reader; the tables are dealt in index order to
    /// the stage holding the fewest bytes.
    /// </summary>
    [Theory]
    [InlineData(2, new[] { 0, 1 }, false)]
    [InlineData(3, new[] { 0, 1 }, false)]
    [InlineData(5, new[] { 0, 1 }, true)]
    public void TheFixtureTables_MoveOnlyWhereOneStageReadsBoth(int cut, int[] owners, bool moves)
    {
        Dsv4Topology t = FixtureTopology();
        int[] device = Dsv4StagePlacement.FromBoundaries(new[] { 0, cut, Dsv41SyntheticModelBuilder.Layers }, t.NLayer);
        Assert.Equal(owners, Dsv4StagePlacement.EngramOwners(t, device, 1, 2, new long[] { 100, 100 }, "balance"));
        // Unset keeps every table with its reader.
        Assert.Equal(new[] { device[1], device[3] }, Dsv4StagePlacement.EngramOwners(t, device, 1, 2, new long[] { 100, 100 }, null));
        int reader1 = device[3];
        Assert.Equal(moves, owners[1] != reader1);
        // Three stages, three equal tables read on stage 0: one each.
        var three = Topology(6, 1024, 16, 64, 1024, 16, 256, 4, new[] { 0, 0, 2, 2, 1, 1 }, new[] { 2, 4 }, new[] { 2, 4, 5 },
            new[] { 0, 1, 2 });
        int[] cut3 = Dsv4StagePlacement.FromBoundaries(new[] { 0, 3, 4, 6 }, 6);
        Assert.Equal(new[] { 0, 1, 2 }, Dsv4StagePlacement.EngramOwners(three, cut3, 1, 3, new long[] { 5, 5, 5 }, "balance"));
        Assert.Equal(new[] { 0, 1, 2 }, Dsv4StagePlacement.EngramOwners(three, cut3, 1, 3, new long[] { 9, 5, 5 }, "balance"));
        // Two stages, three equal tables read on stage 0: the third ties and stays with its reader.
        int[] cut2 = Dsv4StagePlacement.FromBoundaries(new[] { 0, 4, 6 }, 6);
        Assert.Equal(new[] { 0, 1, 0 }, Dsv4StagePlacement.EngramOwners(three, cut2, 1, 2, new long[] { 5, 5, 5 }, "balance"));
        // A small first table: the big second one goes to the emptier stage, the third back to the reader.
        Assert.Equal(new[] { 0, 1, 0 }, Dsv4StagePlacement.EngramOwners(three, cut2, 1, 2, new long[] { 1, 9, 1 }, "balance"));
    }

    // ================================================================ GPU: in-process two-node loopback

    /// <summary>
    /// At the cut after layer 4 both Engram layers run on the worker. With TS_DSV41_ENGRAM_OWNER=balance the driver keeps
    /// table 1 (it maps it, the worker does not) and sends its rows at the start of every forward: one message per
    /// forward, worker-bound only. Every logit vector of the script is bit-identical to the single-process engine's, and
    /// to the same split with the owner unset (the default) and 'local' (the S9 placement: the worker maps both, nothing
    /// crosses). The balanced run keeps no message buffer past its exchange and shrinks the received rows to the last
    /// forward's (both thresholds lowered to 0 here, so every forward reallocates), with the same logits.
    /// </summary>
    [CudaFact]
    public async Task AnEngramTableKeptByTheOtherNode_AnswersToTheBit()
    {
        float[][] reference = Reference();
        int forwards = Calls.Length + 1;   // + the warm-up's one-token forward

        _env.Set("TS_DSV41_ENGRAM_OWNER", "balance");
        long wireKeep = Dsv4CudaEngine.EngramWireKeepBytes, rowsKeep = DeepSeek4CudaExecutor.RemoteRowsKeepBytes;
        Dsv4CudaEngine.EngramWireKeepBytes = 0;
        DeepSeek4CudaExecutor.RemoteRowsKeepBytes = 0;
        Run moved;
        try
        {
            moved = await Distributed(5);
        }
        finally
        {
            Dsv4CudaEngine.EngramWireKeepBytes = wireKeep;
            DeepSeek4CudaExecutor.RemoteRowsKeepBytes = rowsKeep;
        }
        AssertBitIdentical("[engram kept by the driver] distributed vs single-process", reference, moved.Logits);
        StageCounters w = moved.Worker, d = moved.Driver;
        Assert.Equal(new[] { 0, 1 }, w.EngramOwners);
        Assert.Equal(new[] { 0, 1 }, d.EngramOwners);
        Assert.Equal(new[] { true, false }, w.MapsEngram);
        Assert.Equal(new[] { false, true }, d.MapsEngram);
        Assert.Equal(forwards, d.EngramSent);
        Assert.Equal(forwards, w.EngramReceived);
        Assert.Equal(0, d.EngramReceived);
        Assert.Equal(0, w.EngramSent);
        Assert.Equal(d.EngramBytesSent, w.EngramBytesReceived);
        // ids (4 B) + one Q5_K row (176 B) per (token, column), plus the 32-byte header per message.
        long tokens = Calls.Sum(c => (long)c.Tokens.Length) + 1;
        int columns = (Dsv41SyntheticModelBuilder.EngramNgram - 1) * Dsv41SyntheticModelBuilder.EngramHeads;
        Assert.Equal(tokens * columns * (4 + 176) + 32L * forwards, d.EngramBytesSent);
        Assert.Equal(new long[] { Ubatches().Count() + 1, Ubatches().Count() + 1 }, w.EngramRuns);
        // No message buffer outlives its exchange; the worker holds the last forward's rows (one decoded token) only.
        Assert.Equal(0, w.EngramWireBytes);
        Assert.Equal(0, d.EngramWireBytes);
        Assert.Equal(Calls[^1].Tokens.Length * columns * (4L + 176), w.RemoteRowsBytes);

        foreach (string local in new[] { null, "local" })
        {
            _env.Set("TS_DSV41_ENGRAM_OWNER", local);
            Run kept = await Distributed(5);
            string what = local ?? "unset";
            AssertBitIdentical($"[engram kept by its reader, owner {what}] distributed vs single-process", reference, kept.Logits);
            Assert.Equal(new[] { 0, 0 }, kept.Worker.EngramOwners);
            Assert.Equal(new[] { true, true }, kept.Worker.MapsEngram);
            Assert.Equal(new[] { false, false }, kept.Driver.MapsEngram);
            Assert.Equal(0, kept.Driver.EngramSent + kept.Worker.EngramReceived);
            Assert.Equal(0, kept.Worker.RemoteRowsBytes);
        }
        _output.WriteLine($"[engram ownership] {forwards} messages, {d.EngramBytesSent} B driver -> worker; " +
            $"boundary send {w.SendMs:F1} ms, receive {d.ReceiveMs:F1} ms");
    }

    /// <summary>
    /// The ownership is part of the startup digest: a node whose TS_DSV41_ENGRAM_OWNER differs (here 'balance' on one
    /// node, unset on the other) would expect rows its peer never sends, so both refuse at startup, before the executor
    /// opens a shard. The digest compares the mode by meaning, so unset and 'local' agree
    /// (<see cref="TheEngramOwnerDigest_ReadsUnsetAsLocal"/>).
    /// </summary>
    [CudaFact]
    public async Task ADifferentEngramOwnerOnOneNode_IsRefusedAtStartup()
    {
        DeepSeek4Model.DistributedEnvironmentForTests = (node, name) => name != "TS_DSV41_ENGRAM_OWNER"
            ? Environment.GetEnvironmentVariable(name)
            : node == 1 ? "balance" : null;
        string path = Dsv41SyntheticModelBuilder.Fixture();
        IPEndPoint[] endpoints = Endpoints(2);
        var groups = new ITensorParallelGroup[2];
        try
        {
            Exception[] errors = await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
            {
                var group = new DistributedLayerSplitGroup(ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1), node, endpoints);
                groups[node] = group;
                return Record.Exception(() =>
                {
                    using var model = ModelBase.Create(path, BackendType.Cuda, 1, group, layerSplitDegree: 1);
                });
            }))).WaitAsync(LoadDeadline);
            Assert.All(errors, error =>
            {
                var refusal = Assert.IsType<InvalidOperationException>(error);
                Assert.Contains("different model metadata or execution settings", refusal.Message);
                Assert.DoesNotContain("DeepSeek4CudaExecutor..ctor", refusal.StackTrace);
            });
        }
        finally
        {
            DeepSeek4Model.DistributedEnvironmentForTests = null;
            foreach (var group in groups)
                group?.Dispose();
        }
    }

    /// <summary>
    /// A driver that fails while it gathers the rows it keeps for the worker (here: just before the prompt's message goes
    /// out) aborts the mesh: the worker, waiting for those rows, leaves its loop, the driver's forward throws at once, and
    /// every later operation refuses until the nodes are restarted.
    /// </summary>
    [CudaFact]
    public async Task AnOwnerFailingBeforeItsRows_FailsBothNodesWithinTheDeadline()
    {
        _env.Set("TS_DSV41_ENGRAM_OWNER", "balance");
        Dsv4CudaEngine.BeforeEngramRowsSend = (table, sequence) =>
        {
            if (sequence == 1)   // the warm-up sent message 0; message 1 is the prompt's
                throw new IOException("injected owner failure before the prompt's Engram rows");
        };
        Exception failure = null, later = null;
        double seconds = 0;
        try
        {
            await Distributed(5, async driver =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                failure = await Record.ExceptionAsync(() => Task.Run(() => driver.Forward(Prompt)).WaitAsync(TimeSpan.FromSeconds(60)));
                seconds = sw.Elapsed.TotalSeconds;
                later = Record.Exception(() => driver.Forward(new[] { 1 }));
                return Array.Empty<float[]>();
            });
        }
        catch (Exception ex) when (failure != null)
        {
            _output.WriteLine($"teardown after the failure: {ex.GetType().Name}");
        }
        finally
        {
            Dsv4CudaEngine.BeforeEngramRowsSend = null;
        }
        Assert.NotNull(failure);
        Assert.Contains("injected owner failure", failure.ToString());
        Assert.True(seconds < 30, $"the driver took {seconds:F1} s to fail");
        Assert.NotNull(later);
        Assert.Contains("restart every node", later.ToString(), StringComparison.OrdinalIgnoreCase);
    }
}
