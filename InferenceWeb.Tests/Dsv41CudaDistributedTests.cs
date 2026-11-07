// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Cross-node layer split inside the direct-CUDA DeepSeek engine (Dsv4CudaEngine.Distributed.cs).
//
// The wire format and the placement are host code and are tested without a GPU: a boundary payload
// round-trips every section, a payload from another layout or position is refused before its body is read,
// the committed ranges a receiver recomputes match the engine's own arithmetic across ubatch edges, and the
// placement computed from the real DeepSeek-V4.1-Flash-1017 header bytes (a copy in Fixtures/DeepSeek41)
// is the 2 nodes x 2 GPUs cut (12, 22, 31) that the design derived, with no free-VRAM input.
//
// On the GPU, two "nodes" run in this process over real TCP on 127.0.0.1, each a stage on the same CUDA
// device, on the 1017-shaped synthetic fixture (Dsv41SyntheticModelBuilder: ratios [0,0,2,2,1,1], kv sources
// {2,4}, index sources {2,4,5}, candidate source 4, Engram on layers 1 and 3). The cut is forced at three
// places: after layer 2 (kv source 2's rows and its sparse selection cross into layer 3), after layer 4
// (candidate source 4's mask and kv source 4's rows cross into index source 5) and after layer 1 (the Engram
// layer 1's residual crosses as the hidden streams). At each, every logit vector of a multi-ubatch prefill,
// 18 decode steps, a multi-row append and a reset + re-prefill is bit-identical to the single-process engine
// and to a second distributed run, and the boundary counters prove the cut carried what it is there to carry.
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Distributed;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed partial class Dsv41CudaDistributedTests : IDisposable
{
    private const int Context = 1024, Ubatch = 64;
    private static readonly TimeSpan LoadDeadline = TimeSpan.FromSeconds(180), RunDeadline = TimeSpan.FromSeconds(240);

    private readonly ITestOutputHelper _output;
    private readonly NativeEnvScope _env = new();
    private readonly EnvScope _speculation = new();

    public Dsv41CudaDistributedTests(ITestOutputHelper output)
    {
        _output = output;
        _env.Set("MAX_CONTEXT", Context.ToString());
        _env.Set("TS_DSV4_UBATCH", Ubatch.ToString());
        _env.Set("TS_DSV41_YOCO_KEEP", "0");
        // Host mappings on both sides: a distributed stage keeps the tables host-mapped unless =1 asks for
        // VRAM, so the single-process control asks for the same.
        _env.Set("TS_DSV41_ENGRAM_DEVICE", "0");
        foreach (string name in new[] { "TS_DSV4_MMAP", "TS_DSV4_CUDA_TRACE_DIR", "TS_DSV4_PERF", "KV_CACHE_DTYPE", "TS_DSV4_ASYNC_SEND",
                     "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "TENSORSHARP_TP_NODE_ID",
                     "TENSORSHARP_TP_PEERS", LayerSplitPlan.DriverStageVariable, "TS_N_CPU_MOE", "TS_CPU_MOE",
                     "TS_DSV41_ENGRAM_OWNER", "TS_DSV41_ENGRAM_THREADS", "TS_DSV41_ENGRAM_PREFETCH", "TS_DSV41_ENGRAM_RANDOM" })
            _env.Set(name, null);
        _speculation.ClearSpeculationVars();
    }

    public void Dispose()
    {
        DeepSeek4CudaExecutor.ForceStageBoundaries = null;
        DeepSeek4Model.DistributedEnvironmentForTests = null;
        Dsv4CudaEngine.BeforeBoundarySend = null;
        Dsv4CudaEngine.TamperBoundary = null;
        Dsv4CudaEngine.BeforeBoundaryCopies = null;
        Dsv4CudaEngine.FreeVramForTests = null;
        Dsv4CudaEngine.BeforeEngramRowsSend = null;
        Dsv4CudaEngine.LogicalDevicesForTests = 0;
        Dsv4CudaEngine.KernelModuleLoader = null;
        _speculation.Dispose();
        _env.Dispose();
    }

    // ================================================================ topologies (no GPU)

    /// <summary>The synthetic fixture's topology, as the executor derives it from the GGUF metadata.</summary>
    private static Dsv4Topology FixtureTopology(int nCtx = Context, int nUbatch = Ubatch)
    {
        int n = Dsv41SyntheticModelBuilder.Layers;
        return Topology(n, Dsv41SyntheticModelBuilder.Hidden, Dsv41SyntheticModelBuilder.IndexerTopK, nUbatch, nCtx,
            Dsv41SyntheticModelBuilder.SlidingWindow, Dsv41SyntheticModelBuilder.Vocab, Dsv41SyntheticModelBuilder.CandidateSource,
            Dsv41SyntheticModelBuilder.Ratios, Dsv41SyntheticModelBuilder.KvSources, Dsv41SyntheticModelBuilder.IndexSources,
            Dsv41SyntheticModelBuilder.EngramLayers);
    }

    /// <summary>Per layer: the latest kv / index source at or before it with the layer's ratio (LoadEngramMetadata's rule).</summary>
    private static Dsv4Topology Topology(int n, int embd, int topK, int nUbatch, int nCtx, int swa, int vocab, int candidate,
        int[] ratios, int[] kvSources, int[] indexSources, int[] engramLayers)
    {
        var t = new Dsv4Topology
        {
            NLayer = n, NEmbd = embd, HeadDim = 512, IdxHeadSize = 128, IdxTopK = topK, NUbatch = nUbatch, NCtx = nCtx,
            NSwa = swa, NVocab = vocab, CandidateSource = candidate, V41 = true,
            Ratio = ratios.Take(n).ToArray(), KvSource = new int[n], IndexSource = new int[n], EngramIndex = new int[n],
        };
        int kv = -1, index = -1;
        for (int il = 0; il < n; il++)
        {
            if (kvSources.Contains(il)) kv = il;
            if (indexSources.Contains(il)) index = il;
            t.KvSource[il] = t.Ratio[il] != 0 ? kv : -1;
            t.IndexSource[il] = t.Ratio[il] != 0 ? index : -1;
            t.EngramIndex[il] = Array.IndexOf(engramLayers, il);
        }
        return t;
    }

    private sealed record Layout1017(Dsv4Topology Topology, long[] LayerBytes, long Embedding, long Head);

    /// <summary>The real DeepSeek-V4.1-Flash-1017 UD-IQ2_XXS layout (Fixtures/DeepSeek41/layout-1017.json) at the
    /// 2x2 deployment's settings: ubatch 256, 64K context.</summary>
    private static Layout1017 Load1017(int nCtx = 65536, int nUbatch = 256)
    {
        using var stream = typeof(Dsv41CudaDistributedTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.DeepSeek41.layout-1017.json")!;
        using var doc = JsonDocument.Parse(stream);
        JsonElement r = doc.RootElement;
        int[] Ints(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int n = r.GetProperty("block_count").GetInt32();
        var t = Topology(n, r.GetProperty("embedding_length").GetInt32(), r.GetProperty("indexer_top_k").GetInt32(), nUbatch,
            nCtx, r.GetProperty("sliding_window").GetInt32(), r.GetProperty("vocab_size").GetInt32(),
            r.GetProperty("candidate_source_layer").GetInt32(), Ints("compress_ratios"), Ints("kv_source_layers"),
            Ints("indexer_source_layers"), Ints("engram_layers"));
        long Align(long v) => (v + 255) & ~255L;
        long[] bytes = r.GetProperty("layer_bytes").EnumerateArray().Select(e => e.GetInt64()).ToArray();
        return new Layout1017(t, bytes, Align(r.GetProperty("token_embd_bytes").GetInt64()),
            Align(r.GetProperty("output_bytes").GetInt64()) + Align(r.GetProperty("output_norm_bytes").GetInt64()));
    }

    // ================================================================ placement (no GPU)

    /// <summary>
    /// Every node computes the cut from the header bytes alone; for the 1017 on 2 nodes x 2 GPUs (driver last,
    /// the head on node 0) it is (12, 22, 31): node 1 holds layers 0-21 and the embedding, node 0 layers 22-39 and
    /// the head. The plan reads no device at all, so it cannot depend on how much VRAM a node has free.
    /// </summary>
    [Fact]
    public void The1017Cut_From_HeaderBytes_Is_12_22_31()
    {
        Layout1017 l = Load1017();
        foreach (bool driverLast in new[] { true, false })
        {
            int[] device = Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 2, 2, driverLast);
            Assert.Equal(new[] { 0, 12, 22, 31, 40 }, Dsv4StagePlacement.Boundaries(device, 4));
            // The same plan from each node's own view of the pipeline.
            for (int node = 0; node < 2; node++)
            {
                var plan = LayerSplitPlan.CreateWeighted(
                    Dsv4StagePlacement.LayerBytes(l.Topology, l.LayerBytes, device), 2, node, 2, l.Embedding, l.Head, driverLast);
                for (int il = plan.FirstLayer; il < plan.EndLayer; il++)
                    Assert.Equal(device[il], plan.StageIndex * 2 + plan.LocalRank(il));
            }
        }
        var driver = new LayerSplitPlan(40, 2, 0, 2, true);
        Assert.True(driver.IsLast);   // node 0 projects: the logits stay on it
        // Without the replica bytes the cut is the same: replicas are tens of MiB against 3-4.8 GB layers.
        Assert.Equal(new[] { 0, 12, 22, 31, 40 }, Dsv4StagePlacement.Boundaries(Dsv4StagePlacement.Assign(
            Dsv4StagePlacement.LayerBytes(l.Topology, l.LayerBytes, null), l.Embedding, l.Head, 2, 2, true), 4));
        // The 4-way minimax over one node's four GPUs agrees.
        Assert.Equal(new[] { 0, 12, 22, 31, 40 }, Dsv4StagePlacement.Boundaries(
            Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 1, 4, false), 4));

        int[] cut = Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 2, 2, true);
        long[] priced = Dsv4StagePlacement.LayerBytes(l.Topology, l.LayerBytes, cut);
        var perDevice = new double[4];
        for (int il = 0; il < 40; il++)
            perDevice[cut[il]] += priced[il];
        perDevice[0] += l.Embedding;
        perDevice[3] += l.Head;
        _output.WriteLine("[1017 placement] GiB per global device incl. caches and replicas at ctx 65536 / ubatch 256: " +
            string.Join(" / ", perDevice.Select(b => (b / (1L << 30)).ToString("F2"))));
        // Group 8 (ratio 2, layers 8-13) spans node 1's two GPUs; group 20 (ratio 1, layers 20-39) is the only one
        // whose readers are on the other node, so its rows are the only cache rows that cross the network.
        int[] last = TopologyGroupLast(l.Topology, cut);
        Assert.Equal(new[] { 8, 20 }, Enumerable.Range(0, 40).Where(s => last[s] > cut[s]).ToArray());
        Assert.Equal(new[] { 20 }, Enumerable.Range(0, 40).Where(s => last[s] >= 0 && last[s] / 2 > cut[s] / 2).ToArray());
    }

    private static int[] TopologyGroupLast(Dsv4Topology t, int[] device)
    {
        var last = Enumerable.Repeat(-1, t.NLayer).ToArray();
        for (int il = 0; il < t.NLayer; il++)
            if (t.Ratio[il] != 0)
                last[t.KvSource[il]] = Math.Max(last[t.KvSource[il]], device[il]);
        return last;
    }

    // ================================================================ committed ranges (no GPU)

    /// <summary>
    /// The receiver recomputes the blocks a ubatch commits from (p0, nt, ratio): for ratio 2 a block closes at every
    /// odd position, for ratio 1 at every position. Checked against a plain enumeration over every start and width
    /// up to two ubatches, including the edges the engine's script crosses (64 + 64 + 22, a decode at an even and at
    /// an odd position, a 21-token append from an odd position).
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void CommittedRanges_MatchTheBlocksEachUbatchCloses(int ratio)
    {
        for (int p0 = 0; p0 <= 2 * Ubatch + 40; p0++)
        for (int nt = 1; nt <= 2 * Ubatch; nt++)
        {
            var closed = Enumerable.Range(p0, nt).Where(p => (p + 1) % ratio == 0).Select(p => p / ratio).ToArray();
            var (first, count) = Dsv4BoundaryLayout.CommittedRange(p0, nt, ratio);
            Assert.Equal(closed.Length, count);
            if (count > 0)
                Assert.Equal(closed, Enumerable.Range(first, count).ToArray());
        }
        // The script's ubatch edges, spelled out.
        Assert.Equal(ratio == 2 ? (0, 32) : (0, 64), Dsv4BoundaryLayout.CommittedRange(0, 64, ratio));
        Assert.Equal(ratio == 2 ? (32, 32) : (64, 64), Dsv4BoundaryLayout.CommittedRange(64, 64, ratio));
        Assert.Equal(ratio == 2 ? (64, 11) : (128, 22), Dsv4BoundaryLayout.CommittedRange(128, 22, ratio));
        Assert.Equal(ratio == 2 ? (0, 0) : (150, 1), Dsv4BoundaryLayout.CommittedRange(150, 1, ratio));
        Assert.Equal(ratio == 2 ? (75, 1) : (151, 1), Dsv4BoundaryLayout.CommittedRange(151, 1, ratio));
        Assert.Equal(ratio == 2 ? (82, 11) : (165, 21), Dsv4BoundaryLayout.CommittedRange(165, 21, ratio));
    }

    // ================================================================ wire format (no GPU)

    /// <summary>The forced cuts the GPU tests run, by name: the global device boundaries of 2 nodes x 1 GPU.</summary>
    public static IEnumerable<object[]> Cuts() => new[]
    {
        new object[] { "i-kv-rows-and-selection", 3 },
        new object[] { "ii-candidate-mask", 5 },
        new object[] { "iii-after-engram", 2 },
    };

    /// <summary>
    /// A payload written by one node's layout and read by another node's (built independently from the same
    /// topology and assignment) gets every section back byte for byte: the ranges, the streams, the gates, each
    /// range's compressed and indexer rows, the selection and the mask, at every ubatch shape of the script.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cuts))]
    public void ThePayload_RoundTripsEverySection(string name, int cut)
    {
        int[] device = Dsv4StagePlacement.FromBoundaries(new[] { 0, cut, 6 }, 6);
        var sender = Dsv4BoundaryLayout.Create(FixtureTopology(), device, 1);
        var receiver = Dsv4BoundaryLayout.Create(FixtureTopology(), (int[])device.Clone(), 1);
        Assert.Equal(sender.LayoutHash, receiver.LayoutHash);
        var rng = new Random(cut);
        var kinds = new HashSet<Dsv4SectionKind>();
        long sequence = 0;
        foreach (var (p0, nt) in new[] { (0, 64), (64, 64), (128, 22), (150, 1), (151, 1), (165, 21), (1000, 24) })
        {
            Dsv4CommittedRange[] ranges = sender.ExpectedRanges(p0, nt);
            bool active = sender.CandidateBefore;
            int width = sender.CarriesMask(active) ? sender.MaskWidth(p0, nt) : 0;
            Dsv4Section[] sections = sender.Sections(nt, ranges, active, width);
            long body = Dsv4BoundaryLayout.BodyBytes(sections);
            Assert.Equal(0, body % 4);
            var wire = new byte[body];
            var sent = new Dictionary<int, byte[]>();
            for (int i = 0; i < sections.Length; i++)
            {
                Dsv4Section s = sections[i];
                Assert.Equal(0, s.Offset % 4);
                kinds.Add(s.Kind);
                if (s.Kind == Dsv4SectionKind.Ranges)
                {
                    Dsv4BoundaryLayout.WriteRanges(MemoryMarshal.Cast<byte, int>(wire.AsSpan((int)s.Offset, (int)s.Bytes)), ranges);
                    continue;
                }
                var payload = new byte[s.Bytes];
                rng.NextBytes(payload);
                payload.CopyTo(wire, s.Offset);
                sent[i] = payload;
            }
            var header = new int[Dsv4BoundaryLayout.HeaderWords];
            sender.WriteHeader(header, sequence, p0, nt, active, ranges.Length, width, body);

            var (rxActive, rxBody) = receiver.ValidateHeader(header, sequence, p0, nt);
            Assert.Equal(active, rxActive);
            Assert.Equal(body, rxBody);
            Dsv4CommittedRange[] rxRanges = receiver.ValidateRanges(MemoryMarshal.Cast<byte, int>(wire.AsSpan()), p0, nt);
            Assert.Equal(ranges, rxRanges);
            Dsv4Section[] rxSections = receiver.Sections(nt, rxRanges, rxActive, receiver.CarriesMask(rxActive) ? receiver.MaskWidth(p0, nt) : 0);
            Assert.Equal(sections, rxSections);
            foreach (var (i, payload) in sent)
                Assert.Equal(payload, wire.AsSpan((int)rxSections[i].Offset, (int)rxSections[i].Bytes).ToArray());
            sequence++;
        }
        _output.WriteLine($"[{name}] sections: {string.Join(", ", kinds.OrderBy(k => k))}");
        var expected = cut switch
        {
            3 => new[] { Dsv4SectionKind.Ranges, Dsv4SectionKind.Xs, Dsv4SectionKind.Gates, Dsv4SectionKind.CompK,
                Dsv4SectionKind.LidK, Dsv4SectionKind.TopkIdx, Dsv4SectionKind.TopkCnt },
            5 => new[] { Dsv4SectionKind.Ranges, Dsv4SectionKind.Xs, Dsv4SectionKind.Gates, Dsv4SectionKind.CompK,
                Dsv4SectionKind.LidK, Dsv4SectionKind.Mask },
            _ => new[] { Dsv4SectionKind.Ranges, Dsv4SectionKind.Xs, Dsv4SectionKind.Gates },
        };
        Assert.Equal(expected.OrderBy(k => k), kinds.OrderBy(k => k));
    }

    /// <summary>
    /// A header from another model, assignment, position or protocol, or whose committed ranges are not the ones the
    /// receiver recomputes, throws from validation: the engine validates the header before it reads the body and the
    /// ranges before it queues a single copy, so nothing reaches the device (the GPU test below shows that too).
    /// </summary>
    [Fact]
    public void AMismatchedPayload_IsRefusedByValidation()
    {
        int[] device = Dsv4StagePlacement.FromBoundaries(new[] { 0, 3, 6 }, 6);
        var layout = Dsv4BoundaryLayout.Create(FixtureTopology(), device, 1);
        const int p0 = 64, nt = 64;
        Dsv4CommittedRange[] ranges = layout.ExpectedRanges(p0, nt);
        long body = Dsv4BoundaryLayout.BodyBytes(layout.Sections(nt, ranges, false, 0));
        int[] Header()
        {
            var h = new int[Dsv4BoundaryLayout.HeaderWords];
            layout.WriteHeader(h, 7, p0, nt, false, ranges.Length, 0, body);
            return h;
        }
        layout.ValidateHeader(Header(), 7, p0, nt);
        foreach (var (word, what) in new[] { (0, "not a DSV4"), (1, "version"), (2, "layout hash"), (3, "layout hash"),
                     (4, "boundary"), (5, "sequence"), (6, "positions"), (7, "positions"), (8, "candidate"),
                     (9, "committed ranges"), (10, "mask width"), (11, "body") })
        {
            int[] h = Header();
            h[word] += 1;
            var error = Assert.Throws<InvalidOperationException>(() => layout.ValidateHeader(h, 7, p0, nt));
            Assert.Contains(what, error.Message);
        }
        Assert.Throws<InvalidOperationException>(() => layout.ValidateHeader(Header(), 8, p0, nt));
        Assert.Throws<InvalidOperationException>(() => layout.ValidateHeader(Header(), 7, p0 + 1, nt));

        // Another assignment, or the same assignment of another context, is another layout.
        var otherCut = Dsv4BoundaryLayout.Create(FixtureTopology(), Dsv4StagePlacement.FromBoundaries(new[] { 0, 2, 6 }, 6), 1);
        var otherContext = Dsv4BoundaryLayout.Create(FixtureTopology(nCtx: 2048), device, 1);
        var otherUbatch = Dsv4BoundaryLayout.Create(FixtureTopology(nUbatch: 32), device, 1);
        foreach (var other in new[] { otherCut, otherContext, otherUbatch })
        {
            Assert.NotEqual(layout.LayoutHash, other.LayoutHash);
            Assert.Contains("layout hash", Assert.Throws<InvalidOperationException>(() => other.ValidateHeader(Header(), 7, p0, nt)).Message);
        }

        // Committed ranges one block off, of another source or of another length.
        var words = new int[3 * ranges.Length];
        Dsv4BoundaryLayout.WriteRanges(words, ranges);
        Assert.Equal(ranges, layout.ValidateRanges(words, p0, nt));
        for (int w = 0; w < words.Length; w++)
        {
            int[] bad = (int[])words.Clone();
            bad[w] += 1;
            Assert.Contains("committed range", Assert.Throws<InvalidOperationException>(() => layout.ValidateRanges(bad, p0, nt)).Message);
        }
    }

    /// <summary>
    /// The payload of the 1017's node boundary (21|22: global device 2) at the deployment's dims, as the serializer
    /// lays it out: per prefill ubatch of 256 and per decode step at several depths. Kv source 20's rows (ratio 1,
    /// 1,280 B a token), its selection (513 Int32 a token) and the candidate mask (the visible prefix of 65,537
    /// columns) cross beside the 80 KiB of streams a token.
    /// </summary>
    [Fact]
    public void PayloadBytes_AtThe1017Dims()
    {
        Layout1017 l = Load1017();
        int[] device = Dsv4StagePlacement.Plan(l.Topology, l.LayerBytes, l.Embedding, l.Head, 2, 2, true);
        var layout = Dsv4BoundaryLayout.Create(l.Topology, device, 2);
        Assert.Equal(22, layout.FirstLayer);
        Assert.Equal(new[] { 20 }, layout.KvGroups);
        Assert.True(layout.Selection);
        Assert.True(layout.CandidateMask);
        Assert.True(layout.CandidateBefore);
        long Body(int p0, int nt) => Dsv4BoundaryLayout.BodyBytes(layout.Sections(nt, layout.ExpectedRanges(p0, nt), true, layout.MaskWidth(p0, nt)));
        long Pad4(long v) => (v + 3) & ~3L;
        // Decode at depth d (position d): streams 4 x 5120 F32, gates 4 F32, one ratio-1 row of 512 + 128 F16 and its
        // range words, 512 + 1 selection words, and d + 1 mask columns.
        foreach (int depth in new[] { 2048, 8192, 20480, 65535 })
        {
            long expected = 12 + 81920 + 16 + 1024 + 256 + 2048 + 4 + Pad4(depth + 1);
            Assert.Equal(expected, Body(depth, 1));
            _output.WriteLine($"[1017 boundary 21|22] decode at depth {depth}: {Dsv4BoundaryLayout.HeaderBytes + expected} B " +
                $"(of which mask {depth + 1} B)");
        }
        foreach (int p0 in new[] { 0, 1792, 7936, 65280 })
        {
            long expected = 12 + 256L * 81920 + 256 * 16 + 256 * 1280 + 256 * 2048 + 256 * 4 + 256L * (p0 + 256);
            Assert.Equal(expected, Body(p0, 256));
            _output.WriteLine($"[1017 boundary 21|22] prefill ubatch [{p0}, {p0 + 256}): {Dsv4BoundaryLayout.HeaderBytes + expected} B " +
                $"({(Dsv4BoundaryLayout.HeaderBytes + expected) / 1048576.0:F2} MiB, mask {256L * (p0 + 256) / 1048576.0:F2} MiB)");
        }
        _output.WriteLine($"[1017 boundary 21|22] staging capacity (largest body): {layout.MaxBodyBytes()} B");
    }

    // ================================================================ gates (no GPU)

    /// <summary>
    /// Driver-first ordering would ship the full-vocabulary logits back over the network; the direct-CUDA stages
    /// refuse it, before any weight, device or peer is touched.
    /// </summary>
    [Fact]
    public void DriverFirstOrdering_IsRefusedOnCuda()
    {
        _env.Set(LayerSplitPlan.DriverStageVariable, "first");
        var group = new FakeLayerGroup(1, 2, 0);
        var error = Assert.Throws<NotSupportedException>(() =>
            ModelBase.Create(Dsv41SyntheticModelBuilder.Fixture(), BackendType.Cuda, 1, group, layerSplitDegree: 1));
        Assert.Contains("LAST stage", error.Message);
        Assert.Equal(0, group.Reductions);
    }

    // ================================================================ GPU: in-process two-node loopback

    private static readonly int[] Prompt = Tokens(5001, 150);
    private static readonly int[] DecodeTokens = Tokens(5002, 18);
    private static readonly int[] Append = Tokens(5003, 21);
    private const int DecodesBeforeAppend = 15, Redecodes = 3;

    private static int[] Tokens(int seed, int count)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => rng.Next(0, Dsv41SyntheticModelBuilder.Vocab)).ToArray();
    }

    private readonly record struct Call(string Name, int[] Tokens, bool Reset = false);

    /// <summary>A 150-token prompt (ubatches of 64 + 64 + 22), 15 decode steps, a 21-token append from the odd
    /// position 165, 3 more decode steps, then a reset, the prompt again and 3 decode steps: 24 logit vectors.</summary>
    private static readonly Call[] Calls = BuildCalls();

    private static Call[] BuildCalls()
    {
        var calls = new List<Call> { new("prefill", Prompt) };
        for (int i = 0; i < DecodeTokens.Length; i++)
        {
            if (i == DecodesBeforeAppend)
                calls.Add(new("append", Append));
            calls.Add(new($"decode {i + 1}", new[] { DecodeTokens[i] }));
        }
        calls.Add(new("re-prefill", Prompt, Reset: true));
        for (int i = 0; i < Redecodes; i++)
            calls.Add(new($"re-decode {i + 1}", new[] { DecodeTokens[i] }));
        return calls.ToArray();
    }

    /// <summary>Every ubatch of the script: (p0, nt).</summary>
    private static IEnumerable<(int P0, int Nt)> Ubatches()
    {
        int p = 0;
        foreach (Call call in Calls)
        {
            if (call.Reset)
                p = 0;
            for (int off = 0; off < call.Tokens.Length; off += Ubatch)
                yield return (p + off, Math.Min(Ubatch, call.Tokens.Length - off));
            p += call.Tokens.Length;
        }
    }

    private static float[][] Script(Func<int[], float[]> forward, Action reset)
    {
        var rows = new float[Calls.Length][];
        for (int i = 0; i < Calls.Length; i++)
        {
            if (Calls[i].Reset)
                reset();
            rows[i] = (float[])forward(Calls[i].Tokens).Clone();
        }
        return rows;
    }

    private static readonly object ReferenceLock = new();
    private static float[][] _reference;

    /// <summary>The script on the single-process, single-device engine (once per process).</summary>
    private static float[][] Reference()
    {
        lock (ReferenceLock)
        {
            if (_reference == null)
            {
                using ModelBase model = ModelBase.Create(Dsv41SyntheticModelBuilder.Fixture(), BackendType.Cuda);
                _reference = Script(model.Forward, model.ResetKVCache);
            }
            return _reference;
        }
    }

    /// <summary>What one distributed run returned, and what each stage's engine counted: the first stage (a worker),
    /// the last (the driver, node 0) and every stage in pipeline order.</summary>
    private sealed record Run(float[][] Logits, StageCounters Worker, StageCounters Driver, StageCounters[] Stages);

    private sealed record StageCounters(int First, int End, long Sent, long Received, long RowsSent, long RowsReceived,
        long SelectionsSent, long SelectionsReceived, long MasksSent, long MasksReceived, long MaskBytesReceived,
        int[] SourcesSent, int[] SourcesReceived, long[] LayerRuns, long[] EngramRuns, long BytesSent, long BytesReceived,
        double SendMs, double ReceiveMs, long LocalHandOvers, long LocalRows, long LocalSelections, long LocalMasks,
        Dictionary<int, long> LocalRowsBySource, string[] Prefetched, Dsv4DeviceNeed[] Needs,
        long EngramSent, long EngramReceived, long EngramBytesSent, long EngramBytesReceived, int[] EngramOwners, bool[] MapsEngram, long Deferred,
        int EngramWireBytes, long RemoteRowsBytes);

    private static StageCounters Counters(DeepSeek4Model model) => Counters(model.CudaExecutor);

    private static StageCounters Counters(DeepSeek4CudaExecutor executor)
    {
        Dsv4CudaEngine e = executor.Engine;
        Dsv4BoundaryStats s = e.BoundaryStats;
        double f = 1000.0 / Stopwatch.Frequency;
        return new StageCounters(e.OwnedLayers.First, e.OwnedLayers.End, s.Sent, s.Received, s.CommittedRowsSent,
            s.CommittedRowsReceived, s.SelectionsSent, s.SelectionsReceived, s.MasksSent, s.MasksReceived, s.MaskBytesReceived,
            s.SourcesSent.Order().ToArray(), s.SourcesReceived.Order().ToArray(), e.LayerRuns.ToArray(), e.EngramRuns.ToArray(),
            s.BytesSent, s.BytesReceived, s.SendTicks * f, s.ReceiveTicks * f, s.LocalHandOvers, s.LocalRowsHandedOver,
            s.LocalSelectionsHandedOver, s.LocalMasksHandedOver, new Dictionary<int, long>(s.LocalRowsBySource),
            executor.PrefetchedTensorNames.ToArray(), e.StageNeeds?.ToArray(),
            s.EngramMessagesSent, s.EngramMessagesReceived, s.EngramBytesSent, s.EngramBytesReceived,
            executor.EngramOwners?.ToArray(), Enumerable.Range(0, e.EngramRuns.Length).Select(executor.MapsEngramTable).ToArray(), s.Deferred,
            e.EngramWireBytes, executor.RemoteRowsBytes);
    }

    /// <summary>Two nodes (each one stage on the same CUDA device) over TCP on 127.0.0.1, the cut forced after layer
    /// <paramref name="cut"/> - 1. Both warm up first, as the CLI does; node 0 drives the script.</summary>
    private Task<Run> Distributed(int cut, Func<DeepSeek4Model, Task<float[][]>> drive = null)
        => Distributed(new[] { 0, cut, Dsv41SyntheticModelBuilder.Layers }, Dsv41SyntheticModelBuilder.Fixture(), drive);

    /// <summary>One node per global device boundary range of <paramref name="boundaries"/> (each one stage of one
    /// device, all on the same CUDA device) over TCP on 127.0.0.1. Every node warms up first, as the CLI does; node 0
    /// (the last stage) drives the script and the others run the worker loop.</summary>
    private async Task<Run> Distributed(int[] boundaries, string path, Func<DeepSeek4Model, Task<float[][]>> drive = null)
    {
        int nodes = boundaries.Length - 1;
        DeepSeek4CudaExecutor.ForceStageBoundaries = boundaries;
        IPEndPoint[] endpoints = Endpoints(nodes);
        var groups = new ITensorParallelGroup[nodes];
        DeepSeek4Model[] models = Array.Empty<DeepSeek4Model>();
        Task[] workers = Array.Empty<Task>();
        try
        {
            models = await Task.WhenAll(Enumerable.Range(0, nodes).Select(node => Task.Run(() =>
            {
                NodeOfThisLoad.Value = node;
                var group = new DistributedLayerSplitGroup(ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1), node, endpoints);
                groups[node] = group;
                return (DeepSeek4Model)ModelBase.Create(path, BackendType.Cuda, 1, group, layerSplitDegree: 1);
            }))).WaitAsync(LoadDeadline);
            Assert.All(models, m =>
            {
                Assert.False(m.SupportsKVCacheTruncation);
                Assert.False(m.SupportsPerSequenceFusedForward);
                Assert.True(m.CudaExecutor.Engine.IsDistributedStage);
            });
            await Task.WhenAll(models.Select(m => Task.Run(m.WarmUpKernels))).WaitAsync(LoadDeadline);
            DeepSeek4Model driver = models[0];
            workers = models.Skip(1).Select(m => Task.Run(m.RunDistributedWorkerLoop)).ToArray();
            driver.BeginDistributedDriver();
            float[][] rows = drive != null
                ? await drive(driver).WaitAsync(RunDeadline)
                : await Task.Run(() => Script(driver.Forward, driver.ResetKVCache)).WaitAsync(RunDeadline);
            StageCounters[] stages = models.Select(Counters).OrderBy(c => c.First).ToArray();
            var run = new Run(rows, stages[0], Counters(driver), stages);
            groups[0].BroadcastControl(4, Array.Empty<int>());   // shutdown: every worker leaves its loop
            await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(30));
            return run;
        }
        finally
        {
            DeepSeek4CudaExecutor.ForceStageBoundaries = null;
            foreach (var group in groups)
                (group as IDistributedExecutionGroup)?.Abort(new OperationCanceledException("fixture complete"));
            foreach (Task worker in workers)
            {
                try { await worker.WaitAsync(TimeSpan.FromSeconds(30)); } catch { /* the test's own assertion reports it */ }
            }
            foreach (var model in models)
                model?.Dispose();
            foreach (var group in groups)
                group?.Dispose();
        }
    }

    /// <summary>The node whose model this task is loading (an in-process test hook that must act on one node only,
    /// such as <see cref="Dsv4CudaEngine.KernelModuleLoader"/>, reads it; the load runs on the task that set it).</summary>
    private static readonly AsyncLocal<int> NodeOfThisLoad = new();

    private static void AssertBitIdentical(string what, float[][] expected, float[][] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Length, actual[i].Length);
            for (int j = 0; j < expected[i].Length; j++)
            {
                if (BitConverter.SingleToInt32Bits(expected[i][j]) != BitConverter.SingleToInt32Bits(actual[i][j]))
                    Assert.Fail($"{what}: {Calls[i].Name}, logit {j}: {expected[i][j]:R} expected, {actual[i][j]:R} distributed " +
                        $"(max |d| over the vector {expected[i].Zip(actual[i], (a, b) => MathF.Abs(a - b)).Max():G4})");
            }
        }
    }

    /// <summary>
    /// At each forced cut, every logit vector of the script is bit-identical to the single-process engine's, and a
    /// second distributed run repeats the first to the bit. The counters then show the cut carried what it is there
    /// for: (i) kv source 2's committed rows (exactly the blocks the script's ubatches close) and the selection layer 3
    /// attends over; (ii) the candidate mask index source 5 prunes with, and kv source 4's rows; (iii) nothing but the
    /// streams and gates, the streams already holding the Engram layer 1's output (the worker ran Engram table 0 and the
    /// driver table 1, layer 3's).
    /// </summary>
    [CudaTheory]
    [MemberData(nameof(Cuts))]
    public async Task EveryCut_AnswersAsOneProcess_ToTheBit(string name, int cut)
    {
        float[][] reference = Reference();
        Run first = await Distributed(cut);
        Run second = await Distributed(cut);
        AssertBitIdentical($"[{name}] distributed vs single-process", reference, first.Logits);
        AssertBitIdentical($"[{name}] distributed run 2 vs run 1", first.Logits, second.Logits);

        StageCounters w = first.Worker, d = first.Driver;
        int ubatches = Ubatches().Count() + 1;   // + the warm-up's one-token forward
        Assert.Equal((0, cut), (w.First, w.End));
        Assert.Equal((cut, Dsv41SyntheticModelBuilder.Layers), (d.First, d.End));
        Assert.Equal(ubatches, w.Sent);
        Assert.Equal(ubatches, d.Received);
        Assert.Equal(0, w.Received);
        Assert.Equal(0, d.Sent);
        Assert.Equal(w.BytesSent, d.BytesReceived);
        for (int il = 0; il < Dsv41SyntheticModelBuilder.Layers; il++)
        {
            Assert.Equal(il < cut ? ubatches : 0, w.LayerRuns[il]);
            Assert.Equal(il < cut ? 0 : ubatches, d.LayerRuns[il]);
        }
        Assert.Equal(w.RowsSent, d.RowsReceived);
        Assert.Equal(w.SourcesSent, d.SourcesReceived);
        // Each node read ahead only its own layers' small tensors.
        foreach (StageCounters s in first.Stages)
            AssertPrefetchedOnlyOwned(s);
        long ExpectedRows(int source) => Ubatches().Prepend((P0: 0, Nt: 1)).Sum(u =>
            (long)Dsv4BoundaryLayout.CommittedRange(u.P0, u.Nt, Dsv41SyntheticModelBuilder.Ratios[source]).Count);
        switch (cut)
        {
            case 3:
                Assert.Equal(new[] { 2 }, w.SourcesSent);
                Assert.Equal(ExpectedRows(2), d.RowsReceived);
                Assert.Equal(ubatches, d.SelectionsReceived);
                Assert.Equal(0, d.MasksReceived);
                break;
            case 5:
                Assert.Equal(new[] { 4 }, w.SourcesSent);
                Assert.Equal(ExpectedRows(4), d.RowsReceived);
                Assert.Equal(0, d.SelectionsReceived);
                Assert.Equal(ubatches, d.MasksReceived);
                Assert.True(d.MaskBytesReceived > 0);
                break;
            default:
                Assert.Empty(w.SourcesSent);
                Assert.Equal(0, d.RowsReceived);
                Assert.Equal(0, d.SelectionsReceived);
                Assert.Equal(0, d.MasksReceived);
                Assert.Equal(new long[] { ubatches, 0 }, w.EngramRuns);
                Assert.Equal(new long[] { 0, ubatches }, d.EngramRuns);
                break;
        }
        _output.WriteLine($"[{name}] {ubatches} boundary payloads, {d.BytesReceived} B; rows {d.RowsReceived} of sources " +
            $"[{string.Join(",", d.SourcesReceived)}], selections {d.SelectionsReceived}, masks {d.MasksReceived} ({d.MaskBytesReceived} B); " +
            $"engram runs worker [{string.Join(",", w.EngramRuns)}] driver [{string.Join(",", d.EngramRuns)}]; " +
            $"boundary wall time: send {w.SendMs:F1} ms, receive {d.ReceiveMs:F1} ms");
    }

    /// <summary>
    /// A payload whose layout hash differs, or whose committed range is one block off, is refused by the receiving
    /// node before any of its layers runs for that ubatch: the driver's layers ran the first ubatch of the prompt and
    /// never the second. The failure then ends both nodes (the worker's loop leaves within the deadline).
    /// </summary>
    [CudaTheory]
    [InlineData("layout-hash")]
    [InlineData("committed-range")]
    public async Task AMismatchedPayload_IsRefusedBeforeTheReceiverComputes(string tamper)
    {
        Dsv4CudaEngine.TamperBoundary = (stage, header, body, bytes) =>
        {
            // The worker (stage 0)'s second boundary payload: the prompt's second ubatch (the warm-up sent the first).
            if (stage != 0 || header[5] != 2)
                return;
            if (tamper == "layout-hash")
                header[2] ^= 1;
            else
                Marshal.WriteInt32(body, 4, Marshal.ReadInt32(body, 4) + 1);   // range 0's first block
        };
        long[] layerRuns = null;
        Exception failure = null;
        try
        {
            await Distributed(3, async driver =>
            {
                failure = await Record.ExceptionAsync(() => Task.Run(() => driver.Forward(Prompt)));
                layerRuns = driver.CudaExecutor.Engine.LayerRuns.ToArray();
                return Array.Empty<float[]>();
            });
        }
        catch (Exception ex) when (failure != null)
        {
            // The shutdown control after a failed forward cannot reach the worker: expected.
            _output.WriteLine($"[{tamper}] teardown after the refusal: {ex.GetType().Name}");
        }
        finally
        {
            Dsv4CudaEngine.TamperBoundary = null;
        }
        Assert.NotNull(failure);
        string text = failure.ToString();
        Assert.Contains(tamper == "layout-hash" ? "layout hash" : "committed range", text);
        // The warm-up's forward and the prompt's first ubatch ran on the driver's layers; the refused second did not.
        Assert.Equal(new long[] { 0, 0, 0, 2, 2, 2 }, layerRuns);
    }

    /// <summary>
    /// A worker that fails part way through a forward (here: just before it would send the prompt's second ubatch)
    /// aborts the mesh, and the driver, which is waiting for that payload, throws at once instead of hanging until the
    /// transport deadline. Every later operation on the driver refuses until the nodes are restarted.
    /// </summary>
    [CudaFact]
    public async Task AWorkerFailingMidForward_FailsTheDriverWithinTheDeadline()
    {
        Dsv4CudaEngine.BeforeBoundarySend = (stage, sequence) =>
        {
            if (stage == 0 && sequence == 2)
                throw new IOException("injected worker failure before the prompt's second boundary payload");
        };
        Exception failure = null;
        double seconds = 0;
        Exception later = null;
        try
        {
            await Distributed(3, async driver =>
            {
                var sw = Stopwatch.StartNew();
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
            Dsv4CudaEngine.BeforeBoundarySend = null;
        }
        Assert.NotNull(failure);
        Assert.IsNotType<TimeoutException>(failure);
        Assert.True(seconds < 30, $"the driver took {seconds:F1} s to notice the failed worker");
        Assert.NotNull(later);
        Assert.Contains("restart every node", later.ToString(), StringComparison.OrdinalIgnoreCase);
        _output.WriteLine($"driver failed {seconds * 1000:F0} ms after the worker: {failure.GetType().Name}: {failure.Message}");
    }

    /// <summary>
    /// A node whose TS_DSV4_UBATCH differs would run other kernels on other ubatch shapes: both nodes refuse at
    /// startup, in the configuration digest, before the executor opens a shard or the engine touches a device.
    /// </summary>
    [CudaFact]
    public async Task ADifferentUbatchOnOneNode_IsRefusedAtStartup()
    {
        DeepSeek4Model.DistributedEnvironmentForTests = (node, name) =>
            node == 1 && name == "TS_DSV4_UBATCH" ? "32" : Environment.GetEnvironmentVariable(name);
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
                Assert.DoesNotContain("Dsv4CudaEngine", refusal.StackTrace);
            });
        }
        finally
        {
            DeepSeek4Model.DistributedEnvironmentForTests = null;
            foreach (var group in groups)
                group?.Dispose();
        }
    }

    /// <summary>The direct-CUDA local group carries only the GPU count; the ggml one still refuses cuda.</summary>
    [CudaFact]
    public void TheCudaLocalGroup_CarriesOnlyItsDegree()
    {
        using ITensorParallelGroup local = ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 1);
        var devices = Assert.IsType<DirectCudaLayerSplitDevices>(local);
        Assert.Equal(1, devices.Degree);
        Assert.Equal(1, devices.NodeCount);
        Assert.False(devices.IsActive);
        Assert.Throws<NotSupportedException>(() => devices.GetAllocator(0));
        Assert.Throws<NotSupportedException>(() => ModelBase.CreateGgmlLocalLayerSplitGroup(BackendType.Cuda, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, 9));
        int visible = CudaDevice.GetDeviceCount();
        if (visible < 8)
            Assert.Contains("visible", Assert.Throws<NotSupportedException>(() =>
                ModelBase.CreateLocalLayerSplitGroup(BackendType.Cuda, visible + 1)).Message);
    }

    private static IPEndPoint[] Endpoints(int count)
    {
        var listeners = Enumerable.Range(0, count).Select(_ => new TcpListener(IPAddress.Loopback, 0)).ToArray();
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }

    /// <summary>A layer group with no network: geometry only, and a count of the agreements asked of it.</summary>
    private sealed class FakeLayerGroup(int local, int nodes, int node) : ILayerSplitGroup, INestedTensorParallelGroup
    {
        public int Reductions { get; private set; }
        public int NodeId => node;
        public int Degree => local;
        public int GlobalDegree => local * nodes;
        public int GlobalRankOffset => local * node;
        public int NodeCount => nodes;
        public bool IsActive => false;
        public ITensorParallelGroup LocalGroup => throw new NotSupportedException();
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void CrossNodeAllReduce(float[] buffer, int count) { Reductions++; throw new NotSupportedException(); }
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Send(float[] buffer, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] buffer, int count, int source) => throw new NotSupportedException();
        public void Dispose() { }
    }
}
