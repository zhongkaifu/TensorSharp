// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The cross-node layer split's remaining paths (S5 review fixes):
//  - a stage that RECEIVES a group's committed rows and the candidate mask and hands them on, through an in-node
//    boundary to its next GPU (the 1017's 2 x 2 deployment: node 0 receives group 20 and the mask on global device 2
//    and layers 31-39 on device 3 read them) or through the network to the next node. A synthetic topology whose
//    ratio-1 group is read by four layers (Dsv41SyntheticModelBuilder.Topology.Span) puts a device between the
//    source and a reader that itself received the state; Dsv4CudaEngine.LogicalDevicesForTests runs two "GPUs" per
//    stage on this one-GPU box. The local boundary now checks what it hands over the way the network receiver does;
//  - the single-process split on several (logical) devices, which no one-GPU run reached before;
//  - a receiver whose GPU is behind its host (the staging is reused only after its copies ran);
//  - startup: a stage that does not fit makes every node refuse before any upload, a payload beyond one transport
//    frame is refused from the topology, nodes running different kernel images refuse, and a refused load releases
//    the shards;
//  - the server's startup path driving a direct-CUDA stage end to end.
using System.Collections.Concurrent;
using System.Net;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Distributed;
using TensorSharp.Models;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed partial class Dsv41CudaDistributedTests
{
    private static readonly Dsv41SyntheticModelBuilder.Topology Span = Dsv41SyntheticModelBuilder.Topology.Span;

    /// <summary>The span fixture's topology at the tests' context and prefill width.</summary>
    private static Dsv4Topology SpanTopology(int nCtx = Context, int nUbatch = Ubatch)
        => Topology(Span.Layers, Dsv41SyntheticModelBuilder.Hidden, Dsv41SyntheticModelBuilder.IndexerTopK, nUbatch, nCtx,
            Dsv41SyntheticModelBuilder.SlidingWindow, Dsv41SyntheticModelBuilder.Vocab, Span.CandidateSource, Span.Ratios,
            Span.KvSources, Span.IndexSources, Span.EngramLayers);

    // ================================================================ the hand-over check (no GPU)

    /// <summary>
    /// What a local boundary hands over is checked like a network payload: per group the layers past it read, exactly
    /// the blocks the ubatch closes (or one per closing row of a batched step, of that row's sequence), in order, and
    /// an active candidate mask when they prune with it. A receiver that lost the rows it received, a range one block
    /// off, another sequence's rows, an extra range or a stale mask flag is refused instead of being read as zeros.
    /// </summary>
    [Fact]
    public void TheHandOverCheck_RefusesWhatTheUbatchDidNotCommit()
    {
        // The span fixture on 2 nodes x 2 GPUs, cut [0,3,5,7,8]: node 0's in-node boundary into global device 3.
        int[] device = Dsv4StagePlacement.FromBoundaries(new[] { 0, 3, 5, 7, 8 }, Span.Layers);
        var b = Dsv4BoundaryLayout.Create(SpanTopology(), device, 3);
        Assert.Equal(new[] { 4 }, b.KvGroups);
        Assert.True(b.CandidateMask);
        Assert.False(b.Selection);
        const string A = "sequence A", B = "sequence B";
        List<(int, string, int, int)> Noted(params (int Source, string Slot, int First, int Count)[] entries) => entries.ToList();
        void Ok(List<(int, string, int, int)> committed, long p0, int nt, string[] slots = null, int[] positions = null, bool active = true)
            => b.ValidateHandOver(committed, A, p0, nt, slots, positions, active);
        void Refused(string what, List<(int, string, int, int)> committed, long p0, int nt, string[] slots = null,
            int[] positions = null, bool active = true)
            => Assert.Contains(what, Assert.Throws<InvalidOperationException>(() =>
                b.ValidateHandOver(committed, A, p0, nt, slots, positions, active)).Message);

        // A prefill ubatch at [64, 128): ratio 1 closes blocks 64..127. Group 2 (ratio 2) is not read past device 3,
        // so its entry is ignored.
        Ok(Noted((2, A, 32, 32), (4, A, 64, 64)), 64, 64);
        Refused("no committed range", Noted((2, A, 32, 32)), 64, 64);
        Refused("blocks 65+64", Noted((4, A, 65, 64)), 64, 64);
        Refused("blocks 64+63", Noted((4, A, 64, 63)), 64, 64);
        Refused("of another sequence", Noted((4, B, 64, 64)), 64, 64);
        Refused("an extra range", Noted((4, A, 64, 64), (4, A, 64, 64)), 64, 64);
        Refused("candidate mask is not active", Noted((4, A, 64, 64)), 64, 64, active: false);
        // A batched decode step: one block per row, each of its own sequence, in row order.
        string[] rows = { A, B };
        int[] positions = { 150, 33 };
        Ok(Noted((4, A, 150, 1), (4, B, 33, 1)), 0, 2, rows, positions);
        Refused("of another sequence", Noted((4, B, 33, 1), (4, A, 150, 1)), 0, 2, rows, positions);
        Refused("no committed range", Noted((4, A, 150, 1)), 0, 2, rows, positions);

        // Ratio 2 at the in-node boundary into device 1 (layer 3 reads group 2): a ubatch at an even position of
        // odd width closes one block fewer than its width says, and a decode step at an even position closes none.
        var b1 = Dsv4BoundaryLayout.Create(SpanTopology(), device, 1);
        Assert.Equal(new[] { 2 }, b1.KvGroups);
        b1.ValidateHandOver(Noted((2, A, 75, 1)), A, 150, 3, null, null, false);
        b1.ValidateHandOver(Noted(), A, 150, 1, null, null, false);
        Assert.Throws<InvalidOperationException>(() => b1.ValidateHandOver(Noted((2, A, 75, 1)), A, 150, 1, null, null, false));
    }

    // ================================================================ the fit check and the wire (no GPU)

    /// <summary>
    /// The fit check prices each device's cache replicas with the function the placement charges them by: on the 1017's
    /// 2 x 2 cut (12, 22, 31) device 1 holds group 8's (ratio 2), devices 2 and 3 group 20's (ratio 1), and together that
    /// is exactly what the placement added to the layers' bytes for the cut.
    /// </summary>
    [Fact]
    public void TheFitCheck_PricesTheReplicasThePlacementCharges()
    {
        Layout1017 l = Load1017();
        int[] cut = Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 2, 2, true);
        long ratio2 = (65536 / 2 + 1) * 1280L, ratio1 = (65536 + 1) * 1280L;
        long[] perDevice = Enumerable.Range(0, 4).Select(g => Dsv4StagePlacement.DeviceReplicaBytes(l.Topology, cut, g)).ToArray();
        Assert.Equal(new[] { 0, ratio2, ratio1, ratio1 }, perDevice);
        long charged = Dsv4StagePlacement.LayerBytes(l.Topology, l.LayerBytes, cut).Sum()
                       - Dsv4StagePlacement.LayerBytes(l.Topology, l.LayerBytes, null).Sum();
        Assert.Equal(charged, perDevice.Sum());
        _output.WriteLine($"[1017 2x2] replica bytes per global device: {string.Join(" / ", perDevice)}");
    }

    /// <summary>
    /// A boundary payload is one transport frame with a 32-bit length. The 1017's node boundary at the shipped 256-row
    /// ubatch fits at every context up to 1M; a 2048-row ubatch at 1M context would carry a 2 GiB candidate mask alone,
    /// and every node refuses it from the topology before reading anything (not a sender at its first deep ubatch).
    /// </summary>
    [Fact]
    public void APayloadBeyondOneFrame_IsRefusedFromTheTopology()
    {
        // The 2 x 2 deployment's cut, whose node boundary 21|22 carries group 20's rows, the selection and the mask.
        int[] cut = Dsv4StagePlacement.FromBoundaries(new[] { 0, 12, 22, 31, 40 }, 40);
        foreach (var (ubatch, ctx) in new[] { (256, 65536), (256, 1048576), (2048, 262144) })
            Dsv4StagePlacement.ValidateWireLimits(Load1017(ctx, ubatch).Topology, cut, 2);
        Layout1017 big = Load1017(1048576, 2048);
        int[] bigCut = cut;
        long body = Dsv4BoundaryLayout.Create(big.Topology, bigCut, 2).MaxBodyBytes();
        Assert.True(body > Dsv4BoundaryLayout.MaxWireBodyBytes);
        var error = Assert.Throws<InvalidOperationException>(() => Dsv4StagePlacement.ValidateWireLimits(big.Topology, bigCut, 2));
        Assert.Contains("global device 2", error.Message);
        Assert.Contains("TS_DSV4_UBATCH", error.Message);
        // The in-node boundaries (global devices 1 and 3) are not network frames: the same cut on one node of four GPUs passes.
        Dsv4StagePlacement.ValidateWireLimits(big.Topology, bigCut, 4);
        _output.WriteLine($"[1017 2x2] ubatch 2048 at 1M context: {body:N0} B per payload, frame limit {Dsv4BoundaryLayout.MaxWireBodyBytes:N0} B");
    }

    // ================================================================ GPU: received state handed on

    private static readonly ConcurrentDictionary<string, float[][]> References = new();

    /// <summary>The script on the single-process, single-device engine for <paramref name="path"/> (once per process).</summary>
    private static float[][] ReferenceOf(string path) => References.GetOrAdd(path, p =>
    {
        using ModelBase model = ModelBase.Create(p, BackendType.Cuda);
        return Script(model.Forward, model.ResetKVCache);
    });

    private static long ExpectedRows(Dsv41SyntheticModelBuilder.Topology t, int source, bool warmUp)
        => (warmUp ? Ubatches().Prepend((P0: 0, Nt: 1)) : Ubatches())
            .Sum(u => (long)Dsv4BoundaryLayout.CommittedRange(u.P0, u.Nt, t.Ratios[source]).Count);

    /// <summary>
    /// A stage that receives a group's committed rows and the candidate mask hands them on as if it had committed them,
    /// and every logit of the script is bit-identical to the single-process engine's (and repeats run to run):
    /// <list type="bullet">
    /// <item>"2x2-in-node": two nodes of two GPUs each (logical devices on this GPU), cut [0,3,5,7,8]. Node 0 (layers 5-7)
    /// receives group 4's rows, the selection and the mask on its first GPU and hands the rows and the mask to its
    /// second, whose layer 7 reads them: the 1017's 2 x 2 path (group 20 at 30|31). Node 1 hands its own group 2 rows
    /// and selection over its in-node boundary before sending.</item>
    /// <item>"3x1-network": three nodes of one GPU over TCP, cut [0,5,7,8]. The middle node receives group 4 and the mask
    /// and sends them on to the last.</item>
    /// </list>
    /// The counters prove each hand-on carried what the layers past it read; the fit check priced the replicas each GPU
    /// holds; every stage read ahead only its own layers' small tensors; and the stages agreed on their kernel image.
    /// </summary>
    [CudaTheory]
    [InlineData("2x2-in-node")]
    [InlineData("3x1-network")]
    public async Task ReceivedState_IsHandedOn_ToTheBit(string shape)
    {
        string path = Dsv41SyntheticModelBuilder.Fixture(Span);
        float[][] reference = ReferenceOf(path);
        long rows4, rows2 = ExpectedRows(Span, 2, warmUp: false);
        if (shape == "2x2-in-node")
        {
            var first = await LocalStages(path, new[] { 0, 3, 5, 7, 8 }, localDevices: 2);
            var second = await LocalStages(path, new[] { 0, 3, 5, 7, 8 }, localDevices: 2);
            AssertBitIdentical("[2x2] distributed vs single-process", reference, first.Logits);
            AssertBitIdentical("[2x2] distributed run 2 vs run 1", first.Logits, second.Logits);
            long ubatches = Ubatches().Count();
            rows4 = ExpectedRows(Span, 4, warmUp: false);
            StageCounters s0 = first.Stages[0], s1 = first.Stages[1];
            Assert.Equal((0, 5, 5, 8), (s0.First, s0.End, s1.First, s1.End));
            for (int il = 0; il < Span.Layers; il++)
            {
                Assert.Equal(il < 5 ? ubatches : 0, s0.LayerRuns[il]);
                Assert.Equal(il < 5 ? 0 : ubatches, s1.LayerRuns[il]);
            }
            // Node 1 (stage 0): its own group 2 rows and selection into its second GPU, then group 4, the selection
            // and the mask over the network.
            Assert.Equal(ubatches, s0.LocalHandOvers);
            Assert.Equal(new Dictionary<int, long> { [2] = rows2 }, s0.LocalRowsBySource);
            Assert.Equal((ubatches, 0L), (s0.LocalSelections, s0.LocalMasks));
            Assert.Equal((ubatches, 0L), (s0.Sent, s0.Received));
            Assert.Equal(new[] { 4 }, s0.SourcesSent);
            Assert.Equal(rows4, s0.RowsSent);
            // Node 0 (stage 1): receives them on its first GPU and hands group 4's rows and the mask to its second.
            Assert.Equal((0L, ubatches), (s1.Sent, s1.Received));
            Assert.Equal((rows4, ubatches, ubatches), (s1.RowsReceived, s1.SelectionsReceived, s1.MasksReceived));
            Assert.Equal(ubatches, s1.LocalHandOvers);
            Assert.Equal(new Dictionary<int, long> { [4] = rows4 }, s1.LocalRowsBySource);
            Assert.Equal((0L, ubatches), (s1.LocalSelections, s1.LocalMasks));

            // The fit check priced each GPU's replicas: group 2's on global device 1, group 4's on devices 2 and 3.
            Dsv4Topology t = SpanTopology();
            int[] device = Dsv4StagePlacement.FromBoundaries(new[] { 0, 3, 5, 7, 8 }, Span.Layers);
            long r2 = (Context / 2 + 1) * 1280L, r4 = (Context + 1) * 1280L;
            Assert.Equal(new[] { 0L, r2 }, s0.Needs.Select(n => n.Replicas));
            Assert.Equal(new[] { r4, r4 }, s1.Needs.Select(n => n.Replicas));
            foreach (var (stage, needs) in new[] { (0, s0.Needs), (1, s1.Needs) })
            {
                for (int d = 0; d < 2; d++)
                {
                    int g = stage * 2 + d;
                    Assert.Equal(Dsv4StagePlacement.DeviceReplicaBytes(t, device, g), needs[d].Replicas);
                    Assert.Equal(Enumerable.Range(0, Span.Layers).Where(il => device[il] == g)
                        .Sum(il => Dsv4StagePlacement.V41LayerCacheBytes(t, il)), needs[d].Caches);
                    Assert.True(needs[d].Weights > 0);
                    Assert.Equal(g is 0 or 3, needs[d].EmbeddingAndHead > 0);
                }
            }
            foreach (StageCounters s in first.Stages)
                AssertPrefetchedOnlyOwned(s);

            // The kernel-module agreement names the image the devices loaded, the build and the wire version; both
            // stages agreed on it, and on the fit verdict.
            string module = Assert.Single(first.Identities[0], i => i.StartsWith("dsv4-cuda-module|", StringComparison.Ordinal));
            Assert.Equal(module, Assert.Single(first.Identities[1], i => i.StartsWith("dsv4-cuda-module|", StringComparison.Ordinal)));
            AssertModuleIdentity(module, localDevices: 2);
            Assert.All(first.Identities, ids => Assert.Contains("dsv4-cuda-fit|ok", ids));
            _output.WriteLine($"[2x2] module identity: {module}");
            _output.WriteLine($"[2x2] stage 0 hand-overs {s0.LocalHandOvers}: rows {Format(s0.LocalRowsBySource)}, selections " +
                $"{s0.LocalSelections}; sent {s0.Sent} ({s0.BytesSent} B, rows {s0.RowsSent} of [{string.Join(",", s0.SourcesSent)}])");
            _output.WriteLine($"[2x2] stage 1 received {s1.Received} ({s1.BytesReceived} B, rows {s1.RowsReceived}, selections " +
                $"{s1.SelectionsReceived}, masks {s1.MasksReceived}); hand-overs {s1.LocalHandOvers}: rows {Format(s1.LocalRowsBySource)}, " +
                $"masks {s1.LocalMasks}");
            _output.WriteLine($"[2x2] replicas per GPU: {string.Join(" / ", s0.Needs.Concat(s1.Needs).Select(n => n.Replicas))} B");
        }
        else
        {
            Run run = await Distributed(new[] { 0, 5, 7, 8 }, path);
            AssertBitIdentical("[3x1] distributed vs single-process", reference, run.Logits);
            long ubatches = Ubatches().Count() + 1;   // + the warm-up's one-token forward
            rows4 = ExpectedRows(Span, 4, warmUp: true);
            StageCounters a = run.Stages[0], m = run.Stages[1], z = run.Stages[2];
            Assert.Equal((0, 5, 5, 7, 7, 8), (a.First, a.End, m.First, m.End, z.First, z.End));
            Assert.Equal((ubatches, 0L, rows4), (a.Sent, a.Received, a.RowsSent));
            Assert.Equal(new[] { 4 }, a.SourcesSent);
            // The middle node receives group 4's rows, the selection and the mask, and sends the rows and the mask on.
            Assert.Equal((ubatches, ubatches), (m.Received, m.Sent));
            Assert.Equal((rows4, rows4), (m.RowsReceived, m.RowsSent));
            Assert.Equal(new[] { 4 }, m.SourcesSent);
            Assert.Equal((ubatches, 0L), (m.SelectionsReceived, m.SelectionsSent));
            Assert.Equal((ubatches, ubatches), (m.MasksReceived, m.MasksSent));
            Assert.Equal((ubatches, rows4, ubatches, 0L), (z.Received, z.RowsReceived, z.MasksReceived, z.SelectionsReceived));
            Assert.All(run.Stages, s => Assert.Equal(0, s.LocalHandOvers));
            foreach (StageCounters s in run.Stages)
                AssertPrefetchedOnlyOwned(s);
            _output.WriteLine($"[3x1] middle node received {m.Received} ({m.BytesReceived} B, rows {m.RowsReceived}, masks " +
                $"{m.MasksReceived}) and sent {m.Sent} ({m.BytesSent} B, rows {m.RowsSent}, masks {m.MasksSent})");
        }
    }

    private static string Format(Dictionary<int, long> bySource)
        => "{" + string.Join(", ", bySource.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}: {kv.Value}")) + "}";

    /// <summary>A stage reads ahead only its own layers' small tensors (and the non-layer ones), its own included.</summary>
    private static void AssertPrefetchedOnlyOwned(StageCounters s)
    {
        Assert.NotEmpty(s.Prefetched);
        foreach (string name in s.Prefetched)
        {
            if (LayerSplitPlan.TryGetLayer(name, out int il))
                Assert.True(il >= s.First && il < s.End, $"the stage of layers {s.First}..{s.End - 1} read {name} ahead");
        }
        for (int il = s.First; il < s.End; il++)
            Assert.Contains($"blk.{il}.attn_norm.weight", s.Prefetched);
    }

    /// <summary>The module identity: the SHA-256 of the image the devices loaded (the fatbin built with the PTX, or the
    /// PTX), the build the stage was handed, the wire version and the decoders.</summary>
    private static void AssertModuleIdentity(string identity, int localDevices)
    {
        var fields = identity.Split('|').Skip(1).Select(f => f.Split('=', 2)).ToDictionary(f => f[0], f => f[1]);
        string ptxPath = CudaKernels.LocatePtxPath("tensorsharp_dsv4_kernels.ptx");
        byte[] ptx = File.ReadAllBytes(ptxPath);
        var images = new List<string> { Convert.ToHexString(SHA256.HashData(ptx)) };
        if (CudaModule.TryReadMatchingFatbin(ptxPath, ptx) is { } fatbin)
            images.Add(Convert.ToHexString(SHA256.HashData(fatbin)));
        Assert.Contains(fields["image"], images);
        Assert.Equal(TestBuild, fields["build"]);
        Assert.Equal(Dsv4BoundaryLayout.Version.ToString(), fields["wire"]);
        Assert.Equal(localDevices.ToString(), fields["local"]);
        Assert.Equal("True", fields["lowbit_iq"]);
    }

    private const string TestBuild = "in-process test build";

    /// <summary>What a run of in-process stages returned: the driver's logits, each stage's counters and the identities
    /// each stage agreed on, in pipeline order.</summary>
    private sealed record LocalRun(float[][] Logits, StageCounters[] Stages, List<string>[] Identities);

    /// <summary>
    /// The script on one executor per node, each a stage of <paramref name="localDevices"/> logical GPUs on this CUDA
    /// device, the payloads through in-process queues and the startup agreements through an in-process vote. Node 0
    /// runs the last stage and returns the logits; every stage runs each forward concurrently, as the worker loop does.
    /// <paramref name="drive"/>, when given, runs instead of the script: it gets the executors (by node) and the
    /// pipeline's forward, and returns the logit rows.
    /// </summary>
    private async Task<LocalRun> LocalStages(string path, int[] boundaries, int localDevices, Func<int, string> buildOf = null,
        Func<DeepSeek4CudaExecutor[], Func<int[], float[]>, float[][]> drive = null)
    {
        int nodes = (boundaries.Length - 1) / localDevices;
        Dsv4CudaEngine.LogicalDevicesForTests = localDevices;
        DeepSeek4CudaExecutor.ForceStageBoundaries = boundaries;
        using var pipe = new InProcessPipeline(nodes);
        var executors = new DeepSeek4CudaExecutor[nodes];
        try
        {
            await Task.WhenAll(Enumerable.Range(0, nodes).Select(node => Task.Run(() =>
            {
                int stage = LayerSplitPlan.StageOf(node, nodes, driverRunsLastStage: true);
                try
                {
                    executors[node] = new DeepSeek4CudaExecutor(path, Context, Ubatch, new DeepSeek4CudaExecutor.DistributedStage
                    {
                        Nodes = nodes, NodeId = node, LocalDevices = localDevices, DriverRunsLastStage = true,
                        Link = pipe.Link(stage), Agree = pipe.Agree(stage), Build = buildOf?.Invoke(node) ?? TestBuild,
                    });
                }
                catch
                {
                    pipe.Abort();
                    throw;
                }
            }))).WaitAsync(LoadDeadline);
            DeepSeek4CudaExecutor driver = executors[0];
            float[] Forward(int[] tokens)
            {
                var logits = new float[driver.VocabSize];
                Task[] stages = executors.Select(ex => Task.Run(() =>
                {
                    try { ex.Forward(tokens, ex == driver ? logits : new float[ex.VocabSize]); }
                    catch { pipe.Abort(); throw; }
                })).ToArray();
                if (!Task.WhenAll(stages).Wait(RunDeadline))
                {
                    pipe.Abort();
                    throw new TimeoutException("a stage's forward did not finish within the deadline");
                }
                return logits;
            }
            void Reset()
            {
                foreach (var ex in executors)
                    ex.Reset();
            }
            float[][] rows = await Task.Run(() => drive != null ? drive(executors, Forward) : Script(Forward, Reset)).WaitAsync(RunDeadline);
            var byStage = executors.OrderBy(e => e.Engine.OwnedLayers.First).ToArray();
            return new LocalRun(rows, byStage.Select(Counters).ToArray(),
                Enumerable.Range(0, nodes).Select(pipe.Identities).ToArray());
        }
        finally
        {
            pipe.Abort();
            foreach (var ex in executors)
                ex?.Dispose();
            DeepSeek4CudaExecutor.ForceStageBoundaries = null;
            Dsv4CudaEngine.LogicalDevicesForTests = 0;
        }
    }

    /// <summary>In-process stage links (one queue per boundary, stage s to s + 1) and an in-process vote standing in for
    /// the cross-node agreement: every stage's n-th identity must equal every other's, or all of them refuse.</summary>
    private sealed class InProcessPipeline(int stages) : IDisposable
    {
        private static readonly TimeSpan Deadline = TimeSpan.FromSeconds(120);
        private readonly BlockingCollection<byte[]>[] _queues =
            Enumerable.Range(0, Math.Max(0, stages - 1)).Select(_ => new BlockingCollection<byte[]>()).ToArray();
        // Stage-to-stage messages (the Engram rows a node keeps for another's layer), one queue per (from, to).
        private readonly BlockingCollection<byte[]>[,] _direct = DirectQueues(stages);

        private static BlockingCollection<byte[]>[,] DirectQueues(int stages)
        {
            var queues = new BlockingCollection<byte[]>[stages, stages];
            for (int a = 0; a < stages; a++)
                for (int b = 0; b < stages; b++)
                    queues[a, b] = new BlockingCollection<byte[]>();
            return queues;
        }
        private readonly CancellationTokenSource _abort = new();
        private readonly object _lock = new();
        private readonly List<string>[] _agreed = Enumerable.Range(0, stages).Select(_ => new List<string>()).ToArray();

        public IDsv4StageLink Link(int stage) => new QueueLink(this, stage);

        public Action<string> Agree(int stage) => identity => Vote(stage, identity);

        public List<string> Identities(int stage)
        {
            lock (_lock)
                return new List<string>(_agreed[stage]);
        }

        public void Abort()
        {
            lock (_lock)
            {
                _abort.Cancel();
                Monitor.PulseAll(_lock);
            }
        }

        private void Vote(int stage, string identity)
        {
            lock (_lock)
            {
                int round = _agreed[stage].Count;
                _agreed[stage].Add(identity);
                Monitor.PulseAll(_lock);
                var deadline = DateTime.UtcNow + Deadline;
                while (_agreed.Any(a => a.Count <= round))
                {
                    _abort.Token.ThrowIfCancellationRequested();
                    if (!Monitor.Wait(_lock, deadline - DateTime.UtcNow > TimeSpan.Zero ? deadline - DateTime.UtcNow : TimeSpan.Zero)
                        && DateTime.UtcNow >= deadline)
                        throw new TimeoutException($"stage {stage}: agreement round {round} timed out");
                }
                if (_agreed.Any(a => a[round] != identity))
                    throw new InvalidOperationException($"the in-process stages disagree in round {round}: " +
                        string.Join(" / ", _agreed.Select(a => a[round])));
            }
        }

        public void Dispose()
        {
            Abort();
            foreach (var queue in _queues)
                queue.Dispose();
            foreach (var queue in _direct)
                queue.Dispose();
            _abort.Dispose();
        }

        private sealed unsafe class QueueLink(InProcessPipeline pipe, int stage) : IDsv4StageLink
        {
            public void SendNext(IntPtr data, int bytes)
            {
                var copy = new byte[bytes];
                Marshal.Copy(data, copy, 0, bytes);
                pipe._queues[stage].Add(copy, pipe._abort.Token);
            }

            public void ReceivePrevious(IntPtr data, int bytes)
            {
                if (!pipe._queues[stage - 1].TryTake(out byte[] payload, (int)Deadline.TotalMilliseconds, pipe._abort.Token))
                    throw new TimeoutException($"stage {stage}: no payload from stage {stage - 1}");
                if (payload.Length != bytes)
                    throw new InvalidOperationException($"stage {stage}: {payload.Length}-byte payload, {bytes} expected");
                Marshal.Copy(payload, 0, data, bytes);
            }

            public void SendToStage(int to, IntPtr data, int bytes)
            {
                var copy = new byte[bytes];
                Marshal.Copy(data, copy, 0, bytes);
                pipe._direct[stage, to].Add(copy, pipe._abort.Token);
            }

            public void ReceiveFromStage(int from, IntPtr data, int bytes)
            {
                if (!pipe._direct[from, stage].TryTake(out byte[] payload, (int)Deadline.TotalMilliseconds, pipe._abort.Token))
                    throw new TimeoutException($"stage {stage}: no message from stage {from}");
                if (payload.Length != bytes)
                    throw new InvalidOperationException($"stage {stage}: {payload.Length}-byte message from stage {from}, {bytes} expected");
                Marshal.Copy(payload, 0, data, bytes);
            }
        }
    }

    // ================================================================ GPU: the single-process split on several devices

    /// <summary>
    /// A single-process split over 2, 3 and 4 logical devices answers bit for bit as one device does: a prompt over two
    /// ubatches, decode steps, a second sequence and a batched step of both (whose hand-overs carry one block per
    /// closing row, each of its own sequence). Until now this path ran only on a host with several GPUs. Both
    /// topologies: the default one, and the span one, whose ratio-1 group a split separates from its readers.
    /// </summary>
    [CudaTheory]
    [InlineData("default")]
    [InlineData("span")]
    public void ALocalSplit_OnLogicalDevices_AnswersAsOneDevice(string topology)
    {
        string path = topology == "span" ? Dsv41SyntheticModelBuilder.Fixture(Span) : Dsv41SyntheticModelBuilder.Fixture();
        int[] prompt = Tokens(31, 45), steps = Tokens(32, 6), other = Tokens(33, 17);
        (List<float[]> Rows, StageCounters Counters, int[] Devices) RunOn(int gpus)
        {
            Dsv4CudaEngine.LogicalDevicesForTests = 4;
            try
            {
                using var ex = new DeepSeek4CudaExecutor(path, maxContext: Context, nUbatch: 32, nGpu: gpus);
                float[] Step(int[] tokens)
                {
                    var logits = new float[ex.VocabSize];
                    ex.Forward(tokens, logits);
                    return logits;
                }
                var rows = new List<float[]> { Step(prompt) };
                foreach (int t in steps)
                    rows.Add(Step(new[] { t }));
                int firstSlot = ex.ActiveSlot;
                int second = ex.SlotAlloc();
                Assert.True(second >= 0 && ex.SetActiveSlot(second));
                rows.Add(Step(other));
                var batched = new float[2 * ex.VocabSize];
                Assert.True(ex.ForwardBatchedDecode(new[] { firstSlot, second }, new[] { 9, 10 },
                    new[] { prompt.Length + steps.Length, other.Length }, batched));
                rows.Add(batched);
                return (rows, Counters(ex), ex.Engine.LayerDevices);
            }
            finally
            {
                Dsv4CudaEngine.LogicalDevicesForTests = 0;
            }
        }

        var single = RunOn(1);
        Assert.Equal(0, single.Counters.LocalHandOvers);
        foreach (int gpus in new[] { 2, 3, 4 })
        {
            var split = RunOn(gpus);
            Assert.Equal(gpus, split.Devices.Max() + 1);
            Assert.Equal(single.Rows.Count, split.Rows.Count);
            for (int i = 0; i < single.Rows.Count; i++)
                Assert.True(MemoryMarshal.AsBytes(single.Rows[i].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(split.Rows[i].AsSpan())),
                    $"[{topology}] {gpus} devices, forward {i}: differs from one device");
            Assert.True(split.Counters.LocalHandOvers > 0);
            _output.WriteLine($"[{topology}] {gpus} logical devices: layers on [{string.Join(",", split.Devices)}], " +
                $"{split.Counters.LocalHandOvers} hand-overs, rows {Format(split.Counters.LocalRowsBySource)}, selections " +
                $"{split.Counters.LocalSelections}, masks {split.Counters.LocalMasks}; bit-identical to one device");
        }
    }

    // ================================================================ GPU: a receiver behind its host

    private const int StallMilliseconds = 300;

    /// <summary>When each stall a test queued on a stream began and ended (Stopwatch ticks, on the host function's thread).</summary>
    private static readonly ConcurrentQueue<(long Start, long End)> Stalls = new();

    [UnmanagedCallersOnly]
    private static void StallStream(IntPtr milliseconds)
    {
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        Thread.Sleep((int)milliseconds);
        Stalls.Enqueue((start, System.Diagnostics.Stopwatch.GetTimestamp()));
    }

    /// <summary>
    /// A receiver whose GPU runs behind its host keeps the pinned staging until the copies out of it have run. The
    /// driver (node 0, the last stage, in this process) has its stream stalled for 300 ms before it copies each of the
    /// prompt's first two payloads out of the staging, while the worker (node 1, the first stage) is
    /// <c>TensorSharp.Cli</c> in its own process: it runs on and sends the next payload, which the driver's host could
    /// then take into a staging its stalled copies have not read yet. (Two stages in one process share a CUDA context,
    /// whose streams this WDDM GPU serializes, so a stalled receiver there stalls its sender too and the race never
    /// forms; across processes it does, as across nodes.) The driver must take each next payload only after its stalled
    /// copies ran, and every logit of the script stays bit-identical to the single-process engine's.
    /// </summary>
    [CudaFact]
    public async Task ABehindReceiver_KeepsItsStagingUntilItsCopiesRan()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        float[][] reference = Reference();
        IPEndPoint[] endpoints = Endpoints(2);
        string dir = Path.Combine(Path.GetTempPath(), "ts-dsv41-behind-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string prompt = Path.Combine(dir, "prompt.txt");
        File.WriteAllText(prompt, "hello");
        var startInfo = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "dotnet", RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false,
            CreateNoWindow = true, WorkingDirectory = dir,
        };
        foreach (string arg in new[] { CliAssembly(), "--model", path, "--backend", "cuda", "--input", prompt, "--max-tokens", "4",
                     "--temperature", "0", "--no-skills", "--log-dir", Path.Combine(dir, "logs"), "--layer-split", "1",
                     "--tp-node-id", "1", "--tp-peers", string.Join(",", endpoints.Select(e => e.ToString())) })
            startInfo.ArgumentList.Add(arg);
        // The worker must agree with this process on every setting the startup digest covers.
        foreach (string name in new[] { "MAX_CONTEXT", "TS_DSV4_UBATCH", "TS_DSV41_YOCO_KEEP", "TS_DSV41_ENGRAM_DEVICE" })
            startInfo.Environment[name] = Environment.GetEnvironmentVariable(name);
        foreach (string name in new[] { "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TENSORSHARP_TP_NODE_ID",
                     "TENSORSHARP_TP_PEERS", LayerSplitPlan.DriverStageVariable, "KV_CACHE_DTYPE", "TS_DSV4_MMAP", "TS_DSV4_PERF" })
            startInfo.Environment.Remove(name);
        var output = new System.Text.StringBuilder();
        using var worker = new System.Diagnostics.Process { StartInfo = startInfo };
        worker.OutputDataReceived += (_, e) => { if (e.Data != null) lock (output) output.AppendLine(e.Data); };
        worker.ErrorDataReceived += (_, e) => { if (e.Data != null) lock (output) output.AppendLine(e.Data); };

        var staged = new ConcurrentDictionary<long, long>();
        Stalls.Clear();
        ITensorParallelGroup group = null;
        DeepSeek4Model driver = null;
        float[][] rows;
        try
        {
            worker.Start();
            worker.BeginOutputReadLine();
            worker.BeginErrorReadLine();
            await Task.Run(() =>
            {
                group = new DistributedLayerSplitGroup(ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1), 0, endpoints);
                driver = (DeepSeek4Model)ModelBase.Create(path, BackendType.Cuda, 1, group, layerSplitDegree: 1);
                driver.WarmUpKernels();
            }).WaitAsync(LoadDeadline);
            Assert.Equal((3, 6), driver.CudaExecutor.Engine.OwnedLayers);
            driver.BeginDistributedDriver();
            Dsv4CudaEngine.BeforeBoundaryCopies = (stage, sequence, stream) =>
            {
                staged[sequence] = System.Diagnostics.Stopwatch.GetTimestamp();
                // The warm-up sent payload 0; the prompt's ubatches are 1, 2 and 3.
                if (sequence is not (1 or 2))
                    return;
                unsafe
                {
                    delegate* unmanaged<IntPtr, void> stall = &StallStream;
                    CudaDriverApi.cuLaunchHostFunc(stream, (IntPtr)stall, StallMilliseconds).ThrowOnError();
                }
            };
            rows = await Task.Run(() => Script(driver.Forward, driver.ResetKVCache)).WaitAsync(RunDeadline);
        }
        finally
        {
            Dsv4CudaEngine.BeforeBoundaryCopies = null;
            driver?.Dispose();   // releases the worker: it leaves its loop and the CLI exits
            group?.Dispose();
            if (!worker.WaitForExit(60_000))
            {
                try { worker.Kill(entireProcessTree: true); } catch { /* already gone */ }
            }
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
        worker.WaitForExit();
        Assert.True(worker.ExitCode == 0, $"the CLI worker exited with {worker.ExitCode}:\n{output}");
        (long Start, long End)[] stalls = Stalls.OrderBy(s => s.Start).ToArray();
        Assert.Equal(2, stalls.Length);
        double Ms(long t) => (t - staged[1]) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        _output.WriteLine($"[behind receiver] ms after payload 1 was staged: stall 1 {Ms(stalls[0].Start):F1}..{Ms(stalls[0].End):F1}, " +
            $"payload 2 staged at {Ms(staged[2]):F1}; stall 2 {Ms(stalls[1].Start):F1}..{Ms(stalls[1].End):F1}, payload 3 staged at " +
            $"{Ms(staged[3]):F1}");
        Exception logits = Record.Exception(() =>
            AssertBitIdentical("[behind receiver] driver with a separate-process worker vs single-process", reference, rows));
        string verdict = logits == null ? "the logits still matched" : "and the logits differ: " + logits.Message;
        // The driver took each next payload into its staging only once its stalled copies had run.
        Assert.True(staged[2] >= stalls[0].End, $"payload 2 was staged at {Ms(staged[2]):F1} ms, before payload 1's copies ran " +
            $"(the stall ended at {Ms(stalls[0].End):F1} ms); {verdict}");
        Assert.True(staged[3] >= stalls[1].End, $"payload 3 was staged at {Ms(staged[3]):F1} ms, before payload 2's copies ran " +
            $"(the stall ended at {Ms(stalls[1].End):F1} ms); {verdict}");
        if (logits != null)
            throw logits;
    }

    /// <summary>TensorSharp.Cli's own build output (it builds into TensorSharp.Cli/bin/).</summary>
    private static string CliAssembly()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        Assert.NotNull(dir);
        string path = Path.Combine(dir.FullName, "TensorSharp.Cli", "bin", "TensorSharp.Cli.dll");
        Assert.True(File.Exists(path), $"Required host build output not found: {path}");
        return path;
    }

    // ================================================================ GPU: startup refusals

    /// <summary>The shards of a fixture can be opened exclusively: nothing in this process still holds them.</summary>
    private static void AssertShardsReleased(string firstShard)
    {
        string dir = Path.GetDirectoryName(firstShard)!;
        string[] shards = Directory.GetFiles(dir, "*.gguf");
        Assert.NotEmpty(shards);
        foreach (string shard in shards)
        {
            try
            {
                using var exclusive = new FileStream(shard, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException e)
            {
                Assert.Fail($"{Path.GetFileName(shard)} is still open after the refused load: {e.Message}");
            }
        }
    }

    /// <summary>Both nodes of a two-node load of <paramref name="path"/>, the cut after layer <paramref name="cut"/> - 1;
    /// what each threw.</summary>
    private async Task<Exception[]> LoadBothNodes(string path, int cut = 3)
    {
        DeepSeek4CudaExecutor.ForceStageBoundaries = new[] { 0, cut, Dsv41SyntheticModelBuilder.Layers };
        IPEndPoint[] endpoints = Endpoints(2);
        var groups = new ITensorParallelGroup[2];
        try
        {
            return await Task.WhenAll(Enumerable.Range(0, 2).Select(node => Task.Run(() =>
            {
                NodeOfThisLoad.Value = node;
                var group = new DistributedLayerSplitGroup(ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1), node, endpoints);
                groups[node] = group;
                return Record.Exception(() =>
                {
                    using var model = ModelBase.Create(path, BackendType.Cuda, 1, group, layerSplitDegree: 1);
                });
            }))).WaitAsync(LoadDeadline);
        }
        finally
        {
            DeepSeek4CudaExecutor.ForceStageBoundaries = null;
            foreach (var group in groups)
                group?.Dispose();
        }
    }

    /// <summary>
    /// A stage that does not fit its GPU refuses with what it needs, and its peer refuses with it at the same point,
    /// before either uploads a weight; before, the peer streamed its whole share and failed later at the first barrier.
    /// Neither refused load keeps the shards open (the executor's shard set, prefetched tensors and mappings go with it).
    /// </summary>
    [CudaFact]
    public async Task AStageThatDoesNotFit_RefusesOnEveryNode_AndReleasesTheShards()
    {
        string path = Dsv41SyntheticModelBuilder.Variant(new Dsv41SyntheticModelBuilder.Options());
        // Node 0 runs the last stage (stage 1): its GPU reports no free memory.
        Dsv4CudaEngine.FreeVramForTests = (stage, device) => stage == 1 ? 0 : -1;
        Exception[] errors;
        try
        {
            errors = await LoadBothNodes(path);
        }
        finally
        {
            Dsv4CudaEngine.FreeVramForTests = null;
        }
        var shortNode = Assert.IsType<InvalidOperationException>(errors[0]);
        Assert.StartsWith("[dsv4-cuda] stage 1, local device 0 (global device 1, layers 3..5) needs", shortNode.Message);
        Assert.Contains("0.00 GiB free", shortNode.Message);
        var peer = Assert.IsType<InvalidOperationException>(errors[1]);
        Assert.StartsWith("[dsv4-cuda] stage 0 fits its GPUs, but another node's stage does not", peer.Message);
        foreach (Exception error in errors)
        {
            Assert.Contains("CheckStageFits", error.StackTrace);
            Assert.DoesNotContain("StreamAll", error.ToString());
        }
        AssertShardsReleased(path);
        _output.WriteLine($"short node: {shortNode.Message}");
        _output.WriteLine($"peer: {peer.Message}");
    }

    /// <summary>
    /// A single-process load the engine refuses (here a kernel module older than the fixture's IQ1_S experts) releases
    /// the shard set the executor opened and prefetched: the files can be opened exclusively at once.
    /// </summary>
    [CudaFact]
    public void ARefusedSingleProcessLoad_ReleasesTheShards()
    {
        string path = Dsv41SyntheticModelBuilder.Variant(new Dsv41SyntheticModelBuilder.Options());
        byte[] stale = Dsv4CudaEngineLoadTests.ModuleWithoutLowBitIqDecoders();
        Dsv4CudaEngine.KernelModuleLoader = () => Dsv4Kernels.FromModule(CudaModule.LoadFromBytes(stale));
        try
        {
            var error = Assert.Throws<NotSupportedException>(() => new DeepSeek4CudaExecutor(path, Context, Ubatch, nGpu: 1));
            Assert.Contains("IQ1_S", error.Message);
        }
        finally
        {
            Dsv4CudaEngine.KernelModuleLoader = null;
        }
        AssertShardsReleased(path);
    }

    /// <summary>
    /// A prefill width and context whose boundary payload exceeds one transport frame (here 2048 rows at 1M context, at
    /// the cut the candidate mask crosses: the mask alone is 2 GiB) is refused on every node right after the placement,
    /// from the topology, before the small tensors are read or the engine touches a device.
    /// </summary>
    [CudaFact]
    public async Task APayloadBeyondOneFrame_IsRefusedOnEveryNode_BeforeTheEngine()
    {
        _env.Set("MAX_CONTEXT", "1048576");
        _env.Set("TS_DSV4_UBATCH", "2048");
        Exception[] errors = await LoadBothNodes(Dsv41SyntheticModelBuilder.Fixture(), cut: 5);
        Assert.All(errors, error =>
        {
            var refusal = Assert.IsType<InvalidOperationException>(error);
            Assert.Contains("more than one transport frame holds", refusal.Message);
            Assert.Contains("prefill width 2048, context 1048576", refusal.Message);
            Assert.Contains("ValidateWireLimits", refusal.StackTrace);
            Assert.DoesNotContain("Dsv4CudaEngine", refusal.StackTrace);
            Assert.DoesNotContain("BeginStreaming", refusal.StackTrace);
        });
        _output.WriteLine(errors[0].Message);
    }

    /// <summary>
    /// Two nodes whose devices loaded different kernel images (here the same kernel source as another build of the
    /// PTX: one more comment line) refuse together at the module agreement, before either uploads a weight. Before, the
    /// agreement compared only the decoders' capability flags, which these images share.
    /// </summary>
    [CudaFact]
    public async Task NodesRunningDifferentKernelImages_RefuseBeforeUploading()
    {
        string ptxPath = CudaKernels.LocatePtxPath("tensorsharp_dsv4_kernels.ptx");
        byte[] other = File.ReadAllBytes(ptxPath).Concat("\n// another build\n"u8.ToArray()).ToArray();
        Dsv4CudaEngine.KernelModuleLoader = () => NodeOfThisLoad.Value == 1
            ? Dsv4Kernels.FromModule(CudaModule.LoadFromBytes(other, identify: true))
            : Dsv4Kernels.Create();
        Exception[] errors;
        try
        {
            errors = await LoadBothNodes(Dsv41SyntheticModelBuilder.Fixture());
        }
        finally
        {
            Dsv4CudaEngine.KernelModuleLoader = null;
        }
        Assert.All(errors, error =>
        {
            var refusal = Assert.IsType<InvalidOperationException>(error);
            Assert.Contains("different model metadata or execution settings", refusal.Message);
            Assert.Contains("Dsv4CudaEngine..ctor", refusal.StackTrace);
            Assert.DoesNotContain("LoadPlaced", refusal.StackTrace);
        });
    }

    /// <summary>
    /// Two nodes of different builds refuse together at the module agreement, before either uploads a weight: two builds
    /// may lay out the payload or compute a layer differently, and only the source revision tells them apart. The model
    /// layer hands every stage its own release and source revision (TensorSharp.Models' informational version; the CUDA
    /// backend assembly carries none of its own).
    /// </summary>
    [CudaFact]
    public async Task NodesOfDifferentBuilds_RefuseBeforeUploading()
    {
        Assert.Equal(System.Diagnostics.FileVersionInfo.GetVersionInfo(typeof(DeepSeek4Model).Assembly.Location).ProductVersion,
            DeepSeek4Model.DirectCudaBuildIdentity);
        Assert.NotEqual("0.0.0.0", DeepSeek4Model.DirectCudaBuildIdentity);
        var error = await Assert.ThrowsAnyAsync<Exception>(() => LocalStages(Dsv41SyntheticModelBuilder.Fixture(),
            new[] { 0, 3, Dsv41SyntheticModelBuilder.Layers }, localDevices: 1, buildOf: node => $"build of node {node}"));
        string text = error.ToString();
        Assert.Contains("the in-process stages disagree", text);
        Assert.Contains("build=build of node 0", text);
        Assert.Contains("Dsv4CudaEngine..ctor", text);
        Assert.DoesNotContain("LoadPlaced", text);
    }

    // ================================================================ GPU: the server's startup path

    /// <summary>
    /// The server's own startup route (the TENSORSHARP_TP_* variables its --layer-split / --tp-node-id / --tp-peers
    /// translate into, DistributedTensorParallel.CreateGroup, StartupModelLoader: load, warm-up, then driver) loads a
    /// direct-CUDA stage of the DeepSeek V4.1 fixture as node 0, with a worker node in this process; a forward through
    /// the server's model crosses the boundary and is bit-identical to the single-process engine's, and disposing the
    /// server's model releases the worker.
    /// </summary>
    [CudaFact]
    public async Task TheServer_LoadsADirectCudaStage_AndDrivesItsWorker()
    {
        string path = Dsv41SyntheticModelBuilder.Fixture();
        float[][] reference = Reference();
        IPEndPoint[] endpoints = Endpoints(2);
        DeepSeek4CudaExecutor.ForceStageBoundaries = new[] { 0, 3, Dsv41SyntheticModelBuilder.Layers };
        _env.Set("TENSORSHARP_TP_NODE_ID", "0");
        _env.Set("TENSORSHARP_TP_PEERS", string.Join(",", endpoints.Select(e => e.ToString())));
        _env.Set("TENSORSHARP_LAYER_SPLIT_DEGREE", "1");
        string dir = Path.Combine(Path.GetTempPath(), "ts-dsv41-server-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        ITensorParallelGroup workerGroup = null;
        DeepSeek4Model worker = null;
        var workerReady = new TaskCompletionSource();
        Task workerTask = Task.Run(() =>
        {
            try
            {
                workerGroup = new DistributedLayerSplitGroup(ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1), 1, endpoints);
                worker = (DeepSeek4Model)ModelBase.Create(path, BackendType.Cuda, 1, workerGroup, layerSplitDegree: 1);
                worker.WarmUpKernels();
                workerReady.SetResult();
                worker.RunDistributedWorkerLoop();
            }
            catch (Exception e)
            {
                workerReady.TrySetException(e);
                throw;
            }
        });
        var svc = new ModelService(NullLogger<ModelService>.Instance)
        {
            TensorParallelGroupFactory = DistributedTensorParallel.CreateGroup,
        };
        bool disposed = false;
        try
        {
            var options = new ServerHostingOptions(
                startupModelPath: path, startupMmProjPath: null, defaultBackend: "cuda",
                supportedBackends: new[] { new BackendOption("cuda", "CUDA") }, defaultMaxTokens: 16, maxTokensPinned: false,
                defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0, defaultVideoHeight: 0, defaultVideoSteps: 0,
                defaultVideoMode: null, uploadDirectory: dir, logDirectory: Path.Combine(dir, "logs"), fileLoggingEnabled: false,
                samplingDefaults: null, prefixCacheEnabled: false);
            await Task.Run(() => StartupModelLoader.LoadIfConfigured(options, svc, "cuda", NullLogger.Instance)).WaitAsync(LoadDeadline);
            await workerReady.Task.WaitAsync(LoadDeadline);
            var driver = Assert.IsAssignableFrom<DeepSeek4Model>(svc.Model);
            Assert.True(driver.CudaExecutor.Engine.IsDistributedStage);
            Assert.Equal((3, 6), driver.CudaExecutor.Engine.OwnedLayers);
            Assert.Equal((0, 3), worker.CudaExecutor.Engine.OwnedLayers);
            float[] logits = await Task.Run(() => driver.Forward(Prompt)).WaitAsync(RunDeadline);
            AssertBitIdentical("[server] the server's driver vs single-process", new[] { reference[0] }, new[] { logits });
            // The warm-up's forward and the prompt's three ubatches crossed the boundary.
            Assert.Equal(4, driver.CudaExecutor.Engine.BoundaryStats.Received);
            Assert.Equal(4, worker.CudaExecutor.Engine.BoundaryStats.Sent);
            disposed = true;
            svc.Dispose();
            await workerTask.WaitAsync(TimeSpan.FromSeconds(30));
            _output.WriteLine("[server] StartupModelLoader loaded node 0's stage (layers 3..5 + head); the worker (layers 0..2 + " +
                "embedding) mirrored the warm-up and the prompt; the prompt's logits are bit-identical; the worker left its loop " +
                "when the server's model was disposed");
        }
        finally
        {
            DeepSeek4CudaExecutor.ForceStageBoundaries = null;
            if (!disposed)
                svc.Dispose();
            (workerGroup as IDistributedExecutionGroup)?.Abort(new OperationCanceledException("fixture complete"));
            try { await workerTask.WaitAsync(TimeSpan.FromSeconds(30)); } catch { /* the assertions above report it */ }
            worker?.Dispose();
            workerGroup?.Dispose();
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { }
        }
    }
}
