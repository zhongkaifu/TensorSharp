// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Cross-node layer split for the direct-CUDA DeepSeek engine: the contract the
// engine and the model layer share.
//
// A distributed run splits the layers into pipeline stages, one per node, and
// each node's stage over its local GPUs. Every node computes the same split
// from the checkpoint header (never from free VRAM, which differs per node), so
// stage ownership and the payload layout cannot desynchronize. Between two
// stages the V4.1 boundary state crosses as one self-describing payload: the
// hidden streams, the delayed hyper-connection gates, the compressed and
// indexer cache rows the stage just committed for every group whose readers are
// past the boundary, the published sparse selection and the candidate mask --
// what the in-process boundary (Dsv4CudaEngine.LayerSplit.cs) moves between two
// local GPUs, and what the native executor moves between two nodes
// (ggml_ops_deepseek_layer_pipeline.inc, ggml_ops_deepseek_layer_transfer.inc).
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// The network between two pipeline stages. Both calls move exactly <c>bytes</c> bytes, a
    /// multiple of 4 (the layer-split transport carries 32-bit words), and block until done. A
    /// transport fault throws; the engine then fails the forward and the caller aborts the mesh.
    /// </summary>
    public interface IDsv4StageLink
    {
        /// <summary>Send to the next pipeline stage.</summary>
        void SendNext(IntPtr data, int bytes);

        /// <summary>Receive from the previous pipeline stage.</summary>
        void ReceivePrevious(IntPtr data, int bytes);

        /// <summary>Send to any other stage (the Engram rows an owning node sends the node whose layer reads them,
        /// <see cref="Dsv4StagePlan.EngramOwnerStage"/>). A link that cannot reach it throws.</summary>
        void SendToStage(int stage, IntPtr data, int bytes)
            => throw new NotSupportedException($"[dsv4-cuda] this stage link cannot reach stage {stage}.");

        /// <summary>Receive from any other stage.</summary>
        void ReceiveFromStage(int stage, IntPtr data, int bytes)
            => throw new NotSupportedException($"[dsv4-cuda] this stage link cannot reach stage {stage}.");
    }

    /// <summary>One node's part of a cross-node layer split.</summary>
    public sealed class Dsv4StagePlan
    {
        /// <summary>Every layer's global device: stage * <see cref="LocalDevices"/> + local rank, so
        /// global devices increase with the layer index.</summary>
        public int[] GlobalDeviceOfLayer { get; init; }

        /// <summary>This node's pipeline position (0 embeds, the last projects).</summary>
        public int Stage { get; init; }

        public int StageCount { get; init; }

        /// <summary>GPUs per node; every node has the same count.</summary>
        public int LocalDevices { get; init; }

        public IDsv4StageLink Link { get; init; }

        /// <summary>Optional cross-node agreement on an identity string. The engine calls it once its
        /// kernel module has loaded, with the module's decoder capabilities: two nodes whose modules
        /// pick different expert kernels would compute different numbers for the same layers.</summary>
        public Action<string> Agree { get; init; }

        /// <summary>The build this node runs (the model layer's release and source revision), part of the module
        /// identity the stages agree on: two builds may lay out the payload or compute the layers differently.</summary>
        public string Build { get; init; }

        /// <summary>
        /// Per Engram table, the stage whose node maps it and reads its rows (null: the stage that runs the table's
        /// layer, as before). A table owned by another stage than its reader's crosses as raw rows at the start of
        /// each forward (<see cref="Dsv4StagePlacement.EngramOwners"/>), so each node keeps only its own tables warm.
        /// Every node computes it from the same placement and agrees on it before loading.
        /// </summary>
        public int[] EngramOwnerStage { get; init; }

        public int FirstLocalGlobalDevice => Stage * LocalDevices;

        /// <summary>Refuses a plan the engine cannot run (not contiguous, a device without layers, the
        /// wrong layer count), before any device is touched.</summary>
        internal void Validate(int nLayer)
        {
            if (GlobalDeviceOfLayer == null || GlobalDeviceOfLayer.Length != nLayer)
                throw new ArgumentException($"[dsv4-cuda] the stage plan names {GlobalDeviceOfLayer?.Length ?? 0} layers; the model has {nLayer}.");
            if (StageCount < 2 || (uint)Stage >= (uint)StageCount || LocalDevices < 1)
                throw new ArgumentException($"[dsv4-cuda] invalid stage {Stage} of {StageCount} with {LocalDevices} local device(s).");
            if (Link == null)
                throw new ArgumentException("[dsv4-cuda] a distributed stage needs a link to its neighbours.");
            int devices = StageCount * LocalDevices;
            var layers = new int[devices];
            for (int il = 0; il < nLayer; il++)
            {
                int g = GlobalDeviceOfLayer[il];
                if ((uint)g >= (uint)devices || (il > 0 && g < GlobalDeviceOfLayer[il - 1]))
                    throw new ArgumentException($"[dsv4-cuda] the stage plan is not a contiguous pipeline at layer {il} (device {g}).");
                layers[g]++;
            }
            for (int g = 0; g < devices; g++)
            {
                if (layers[g] == 0)
                    throw new ArgumentException($"[dsv4-cuda] global device {g} of the stage plan owns no layer.");
            }
        }
    }

    /// <summary>
    /// The model topology the cross-node placement and the boundary payload are derived from: the same on
    /// every node because it comes from the checkpoint header and the execution settings.
    /// </summary>
    public sealed class Dsv4Topology
    {
        public int NLayer, NEmbd, HeadDim, IdxHeadSize, IdxTopK, NUbatch, NCtx, NSwa, NVocab;
        public int CandidateSource = -1;
        public bool V41;
        /// <summary>Per layer: compression ratio, the group source whose caches it reads and the index
        /// source whose selection it attends over (-1 on uncompressed layers), and its Engram table (-1).</summary>
        public int[] Ratio, KvSource, IndexSource, EngramIndex;

        /// <summary>Rows of the widest V4.1 compressed cache (ratio 1), which is also the width of the
        /// indexer scores and of the candidate mask.</summary>
        public int MaskRows => Math.Max(NCtx / 2 + 1, NCtx + 1);

        /// <summary>Rows a V4.1 group of <paramref name="ratio"/> keeps for the whole context.</summary>
        public long V41Rows(int ratio) => ratio == 2 ? NCtx / 2 + 1 : NCtx + 1;

        public static Dsv4Topology From(Dsv4CudaEngine.ModelDesc m)
        {
            var t = new Dsv4Topology
            {
                NLayer = m.NLayer, NEmbd = m.NEmbd, HeadDim = m.HeadDim, IdxHeadSize = m.IdxHeadSize, IdxTopK = m.IdxTopK,
                NUbatch = m.NUbatch, NCtx = m.NCtx, NSwa = m.NSwa, NVocab = m.NVocab, CandidateSource = m.CandidateSource,
                V41 = m.V41,
                Ratio = new int[m.NLayer], KvSource = new int[m.NLayer], IndexSource = new int[m.NLayer], EngramIndex = new int[m.NLayer],
            };
            for (int il = 0; il < m.NLayer; il++)
            {
                Dsv4CudaEngine.LayerDesc l = m.Layers[il];
                t.Ratio[il] = l.Ratio;
                t.KvSource[il] = l.KvSource;
                t.IndexSource[il] = l.IndexSource;
                t.EngramIndex[il] = l.EngramIndex;
            }
            return t;
        }

        /// <summary>Per KV-source layer: the global device of the last layer reading its caches (-1 for a
        /// layer no compressed layer reads).</summary>
        internal int[] GroupLastDevice(int[] globalDevice)
        {
            var last = new int[NLayer];
            Array.Fill(last, -1);
            for (int il = 0; il < NLayer; il++)
            {
                int source = Ratio[il] != 0 ? KvSource[il] : -1;
                if (source >= 0)
                    last[source] = Math.Max(last[source], globalDevice[il]);
            }
            return last;
        }
    }

    /// <summary>
    /// The deterministic cross-node layer placement: header bytes and the execution settings in, every
    /// layer's global device out. No free-VRAM reading anywhere, so every node computes the same answer;
    /// the engine then only checks that its own part fits.
    /// </summary>
    public static class Dsv4StagePlacement
    {
        /// <summary>The per-device run-time bytes of one layer beyond its weights: the raw ring (and its
        /// rewind shadow), and on a ratio group's source layer the group's compressed and indexer caches
        /// and its compressor state ring (and shadow). The engine's own LayerCacheBytes for V4.1.</summary>
        public static long V41LayerCacheBytes(Dsv4Topology t, int il, int maxDraft = 0)
            => V41LayerCacheBytes(t.Ratio[il], t.Ratio[il] != 0 && t.KvSource[il] == il, t.NSwa, t.NUbatch, t.NCtx,
                t.HeadDim, t.IdxHeadSize, maxDraft);

        /// <param name="groupSource">The layer is its ratio group's source and owns the group's caches.</param>
        public static long V41LayerCacheBytes(int ratio, bool groupSource, int nSwa, int nUbatch, int nCtx,
            int headDim, int idxHeadSize, int maxDraft)
        {
            long ring = (nSwa + nUbatch + 255) / 256 * 256;
            long b = ring * headDim * 2 * 2;                                  // RingK + its rewind shadow (F16)
            if (ratio != 0 && groupSource)
            {
                long rows = ratio == 2 ? nCtx / 2 + 1 : nCtx + 1;
                b += rows * headDim * 2 + rows * idxHeadSize * 2;             // CompK + LidK (F16)
                if (ratio > 1)
                    b += 2L * 2 * (ratio + maxDraft) * headDim * 4;           // Hist kv+score and shadows (F32)
            }
            return b;
        }

        /// <summary>Bytes of one replica of a V4.1 group's caches (compressed + indexer rows) on a device
        /// past its source.</summary>
        public static long ReplicaBytes(Dsv4Topology t, int source)
            => t.V41Rows(t.Ratio[source]) * (t.HeadDim + t.IdxHeadSize) * 2L;

        /// <summary>The cache replicas global device <paramref name="g"/> holds: one for every group whose source
        /// lies on an earlier device and whose readers reach <paramref name="g"/>. The engine's fit check prices a
        /// device with this, and <see cref="LayerBytes"/> charges the same replicas to the placement.</summary>
        public static long DeviceReplicaBytes(Dsv4Topology t, int[] globalDevice, int g)
        {
            ArgumentNullException.ThrowIfNull(t);
            ArgumentNullException.ThrowIfNull(globalDevice);
            if (!t.V41)
                return 0;
            int[] last = t.GroupLastDevice(globalDevice);
            long bytes = 0;
            for (int source = 0; source < t.NLayer; source++)
            {
                if (globalDevice[source] < g && g <= last[source])
                    bytes = checked(bytes + ReplicaBytes(t, source));
            }
            return bytes;
        }

        /// <summary>
        /// Refuses a placement whose largest boundary payload cannot cross the wire: a body is one transport frame
        /// with a 32-bit length (<see cref="Dsv4BoundaryLayout.MaxWireBodyBytes"/>), and the receiving staging is
        /// sized for the largest one at load. Checked on every node for every network boundary of the run, from
        /// the topology alone, so all nodes refuse together before any weight is read rather than one sender
        /// failing at its first deep prefill ubatch.
        /// </summary>
        public static void ValidateWireLimits(Dsv4Topology t, int[] globalDevice, int localDevices)
        {
            ArgumentNullException.ThrowIfNull(t);
            ArgumentNullException.ThrowIfNull(globalDevice);
            if (localDevices < 1)
                throw new ArgumentOutOfRangeException(nameof(localDevices));
            int devices = globalDevice.Length == 0 ? 0 : globalDevice[^1] + 1;
            for (int g = localDevices; g < devices; g += localDevices)
            {
                if (Array.IndexOf(globalDevice, g) <= 0)
                    continue;
                long body = Dsv4BoundaryLayout.Create(t, globalDevice, g).MaxBodyBytes();
                if (body > Dsv4BoundaryLayout.MaxWireBodyBytes)
                    throw new InvalidOperationException(
                        $"[dsv4-cuda] the cross-node boundary into global device {g} would carry up to {body:N0} bytes per " +
                        $"ubatch (prefill width {t.NUbatch}, context {t.NCtx}: the candidate mask alone is {t.NUbatch} x " +
                        $"{t.MaskRows} bytes), more than one transport frame holds ({Dsv4BoundaryLayout.MaxWireBodyBytes:N0}). " +
                        "Lower TS_DSV4_UBATCH or MAX_CONTEXT.");
            }
        }

        /// <summary>
        /// Every layer's global device for <paramref name="nodes"/> stages of <paramref name="localDevices"/>
        /// GPUs each, by <see cref="LayerSplitPlan.CreateWeighted"/> over the resident bytes of each layer:
        /// weights + caches, plus the replica bytes of the groups whose readers a cut separates from their
        /// source. A replica is charged to the first layer of the device that holds it, under the cut the
        /// weights and caches alone give, and the plan is cut once more with those bytes (replicas are a
        /// few tens of MiB against GiB-sized layers; the second cut is final either way, so the result is
        /// a pure function of its inputs). The embedding counts on the first device, the head on the last.
        /// </summary>
        public static int[] Plan(Dsv4Topology t, long[] layerWeightBytes, long embeddingBytes, long headBytes,
            int nodes, int localDevices, bool driverRunsLastStage)
        {
            ArgumentNullException.ThrowIfNull(t);
            ArgumentNullException.ThrowIfNull(layerWeightBytes);
            if (layerWeightBytes.Length != t.NLayer)
                throw new ArgumentException("one byte count per layer is required", nameof(layerWeightBytes));
            long[] bytes = LayerBytes(t, layerWeightBytes, null);
            int[] first = Assign(bytes, embeddingBytes, headBytes, nodes, localDevices, driverRunsLastStage);
            if (!t.V41)
                return first;
            return Assign(LayerBytes(t, layerWeightBytes, first), embeddingBytes, headBytes, nodes, localDevices,
                driverRunsLastStage);
        }

        /// <summary>Weights + caches per layer, plus (when <paramref name="globalDevice"/> is given) each
        /// replica that assignment needs, charged to the first layer of its device.</summary>
        public static long[] LayerBytes(Dsv4Topology t, long[] layerWeightBytes, int[] globalDevice)
        {
            var bytes = new long[t.NLayer];
            for (int il = 0; il < t.NLayer; il++)
                bytes[il] = checked(layerWeightBytes[il] + (t.V41 ? V41LayerCacheBytes(t, il) : 0));
            if (globalDevice == null || !t.V41)
                return bytes;
            int[] last = t.GroupLastDevice(globalDevice);
            for (int source = 0; source < t.NLayer; source++)
            {
                if (last[source] <= globalDevice[source])
                    continue;
                for (int g = globalDevice[source] + 1; g <= last[source]; g++)
                {
                    int firstLayer = Array.IndexOf(globalDevice, g);
                    if (firstLayer >= 0)
                        bytes[firstLayer] = checked(bytes[firstLayer] + ReplicaBytes(t, source));
                }
            }
            return bytes;
        }

        /// <summary>The global devices <see cref="LayerSplitPlan.CreateWeighted"/> gives every node.</summary>
        public static int[] Assign(long[] layerBytes, long embeddingBytes, long headBytes,
            int nodes, int localDevices, bool driverRunsLastStage)
        {
            var device = new int[layerBytes.Length];
            Array.Fill(device, -1);
            for (int node = 0; node < nodes; node++)
            {
                var plan = LayerSplitPlan.CreateWeighted(layerBytes, localDevices, node, nodes,
                    embeddingBytes, headBytes, driverRunsLastStage);
                for (int il = plan.FirstLayer; il < plan.EndLayer; il++)
                    device[il] = plan.StageIndex * localDevices + plan.LocalRank(il);
            }
            if (Array.IndexOf(device, -1) >= 0)
                throw new InvalidOperationException("[dsv4-cuda] the layer split left a layer without a device");
            return device;
        }

        /// <summary>The global device boundaries of an assignment: entry g is the first layer of device g,
        /// the last entry the layer count.</summary>
        public static int[] Boundaries(int[] globalDevice, int devices)
        {
            var b = new int[devices + 1];
            for (int g = 0; g < devices; g++)
                b[g] = Array.IndexOf(globalDevice, g);
            b[devices] = globalDevice.Length;
            return b;
        }

        /// <summary>An assignment from device boundaries (<see cref="Boundaries"/>'s form).</summary>
        public static int[] FromBoundaries(int[] boundaries, int nLayer)
        {
            if (boundaries == null || boundaries.Length < 2 || boundaries[0] != 0 || boundaries[^1] != nLayer)
                throw new ArgumentException($"device boundaries must run from 0 to {nLayer}", nameof(boundaries));
            var device = new int[nLayer];
            for (int g = 0; g + 1 < boundaries.Length; g++)
            {
                if (boundaries[g + 1] <= boundaries[g])
                    throw new ArgumentException("every device needs at least one layer", nameof(boundaries));
                for (int il = boundaries[g]; il < boundaries[g + 1]; il++)
                    device[il] = g;
            }
            return device;
        }

        /// <summary>
        /// TS_DSV41_ENGRAM_OWNER as the stages agree on it: unset, empty or <c>local</c> is <c>local</c>, <c>balance</c> is
        /// <c>balance</c> (both trimmed); any other value comes back as given (trimmed), for
        /// <see cref="EngramOwners"/> to refuse. So a node that leaves it unset and one that says <c>local</c> agree.
        /// </summary>
        public static string EngramOwnerModeName(string mode)
        {
            string m = mode?.Trim();
            return string.IsNullOrEmpty(m) ? "local" : m;
        }

        /// <summary>
        /// Which stage keeps each host-mapped Engram table: <c>local</c> (the default, <c>TS_DSV41_ENGRAM_OWNER</c>
        /// unset) leaves every table with the stage whose layer reads it (S9); <c>balance</c> hands the tables out in
        /// index order, each to the stage holding the fewest table bytes so far, keeping it with its reader on a tie.
        /// On the 1017 at (12, 22, 31) both tables (layers 1 and 14) are read on stage 0: with <c>balance</c> table 0
        /// stays there and table 1 goes to stage 1, so each node holds one 63 GiB slice. <c>balance</c> is opt-in until
        /// a two-node run has measured it: it moves every forward's rows of a moved table before the reader's first
        /// layer, and the keeping node does not read them ahead. Device-resident tables never move.
        /// </summary>
        /// <param name="tableBytes">Each table's bytes (from the shard headers, the same on every node).</param>
        /// <param name="mode">null or <c>local</c>, or <c>balance</c>; anything else is refused.</param>
        public static int[] EngramOwners(Dsv4Topology t, int[] globalDevice, int localDevices, int stageCount,
            long[] tableBytes, string mode)
        {
            int tables = 0;
            for (int il = 0; il < t.NLayer; il++)
                tables = Math.Max(tables, t.EngramIndex[il] + 1);
            if (tableBytes == null || tableBytes.Length < tables)
                throw new ArgumentException($"[dsv4-cuda] {tables} Engram tables need their sizes", nameof(tableBytes));
            bool balance = EngramOwnerModeName(mode) switch
            {
                "balance" => true,
                "local" => false,
                _ => throw new NotSupportedException($"TS_DSV41_ENGRAM_OWNER must be 'local' or 'balance' (unset is local), got '{mode}'."),
            };
            var owner = new int[tables];
            var held = new long[stageCount];
            for (int e = 0; e < tables; e++)
            {
                int reader = -1;
                for (int il = 0; il < t.NLayer; il++)
                {
                    if (t.EngramIndex[il] == e)
                        reader = globalDevice[il] / localDevices;
                }
                if (reader < 0)
                    throw new InvalidOperationException($"[dsv4-cuda] no layer reads Engram table {e}");
                int best = reader;
                if (balance)
                {
                    for (int s = 0; s < stageCount; s++)
                    {
                        if (held[s] < held[best])
                            best = s;
                    }
                }
                owner[e] = best;
                held[best] += tableBytes[e];
            }
            return owner;
        }

        /// <summary>FNV-1a over the assignment: the identity every node agrees on before loading.</summary>
        public static ulong AssignmentHash(int[] globalDevice)
        {
            ulong h = Fnv.Offset;
            Fnv.Mix(ref h, (ulong)globalDevice.Length);
            foreach (int g in globalDevice)
                Fnv.Mix(ref h, (ulong)(uint)g);
            return h;
        }
    }

    internal static class Fnv
    {
        public const ulong Offset = 1469598103934665603UL;
        private const ulong Prime = 1099511628211UL;

        public static void Mix(ref ulong h, ulong v)
        {
            for (int i = 0; i < 8; i++)
            {
                h ^= (v >> (8 * i)) & 0xff;
                h *= Prime;
            }
        }
    }

    /// <summary>A run of cache blocks a group's source committed in one ubatch.</summary>
    internal readonly record struct Dsv4CommittedRange(int Source, int FirstBlock, int Count);

    /// <summary>What a payload section holds.</summary>
    internal enum Dsv4SectionKind
    {
        Ranges,     // the committed ranges, (source, first block, count) per entry, as Int32 words
        Xs,         // the hidden streams [nt, hc * n_embd] F32
        Gates,      // the delayed hyper-connection gates [nt, hc] F32
        CompK,      // committed compressed-cache rows [count, head_dim] F16
        LidK,       // committed indexer-cache rows [count, idx_head] F16
        TopkIdx,    // the published selection [nt, idx_top_k] I32
        TopkCnt,    // its counts [nt] I32
        Mask,       // the candidate mask, the first Width columns of [nt, mask_rows] U8
    }

    /// <summary>One section of a boundary payload: where it sits in the body and how big it is. A
    /// committed-row section names its range; the mask section its pitched shape.</summary>
    internal readonly record struct Dsv4Section(Dsv4SectionKind Kind, long Offset, long Bytes, int Range = -1,
        int Rows = 0, int Width = 0);

    /// <summary>
    /// What one boundary carries and how its payload is laid out, built the same way on every node from the
    /// topology and the global assignment. Pure host code, so the wire format is testable without a GPU.
    ///
    /// <para>The payload is two transfers. The header, <see cref="HeaderWords"/> Int32 words: magic "DSV4",
    /// version, the 64-bit layout hash, the boundary's global device, the boundary sequence number, p0, nt,
    /// whether the candidate mask is active, the number of committed ranges, the mask width and the body's
    /// byte count. Then the body: the ranges as (source, first block, count) words, the hidden streams, the
    /// gates, each range's compressed then indexer rows, the selection and its counts, and the mask. Every
    /// section starts 4-byte aligned. The receiver validates the header (and from it knows the body size)
    /// and then recomputes the committed ranges from (p0, nt, ratio): a payload from a node with another
    /// model, assignment or position is refused before anything reaches the device.</para>
    /// </summary>
    internal sealed class Dsv4BoundaryLayout
    {
        public const int Magic = 0x34565344;   // "DSV4", little-endian
        public const int Version = 1;
        public const int HeaderWords = 12;
        public const int HeaderBytes = HeaderWords * 4;
        /// <summary>The largest body one transport frame carries: the frame length is a 32-bit count of the body
        /// and its type byte, and bodies are whole 32-bit words.</summary>
        public const long MaxWireBodyBytes = (int.MaxValue - 8) & ~3L;
        private const int Hc = 4;

        /// <summary>The global device the boundary leads into.</summary>
        public int Boundary { get; }
        /// <summary>The first layer past the boundary.</summary>
        public int FirstLayer { get; }
        /// <summary>KV-source layers before the boundary whose caches layers past it read.</summary>
        public int[] KvGroups { get; }
        /// <summary>A compressed layer past the boundary attends over a selection published before it.</summary>
        public bool Selection { get; }
        /// <summary>An index source past the boundary prunes with the mask published before it.</summary>
        public bool CandidateMask { get; }
        /// <summary>The candidate layer runs before the boundary, so the mask is active when it crosses.</summary>
        public bool CandidateBefore { get; }
        public ulong LayoutHash { get; }

        private readonly Dsv4Topology _t;
        private readonly int[] _maskReaderRatios;

        private Dsv4BoundaryLayout(Dsv4Topology t, int boundary, int firstLayer, int[] kvGroups, bool selection,
            bool candidateMask, bool candidateBefore, int[] maskReaderRatios, int[] globalDevice)
        {
            _t = t;
            Boundary = boundary;
            FirstLayer = firstLayer;
            KvGroups = kvGroups;
            Selection = selection;
            CandidateMask = candidateMask;
            CandidateBefore = candidateBefore;
            _maskReaderRatios = maskReaderRatios;

            ulong h = Fnv.Offset;
            foreach (long v in new long[]
                     {
                         Version, t.V41 ? 1 : 0, t.NLayer, t.NEmbd, Hc, t.HeadDim, t.IdxHeadSize, t.IdxTopK, t.NUbatch,
                         t.NCtx, t.NVocab, t.MaskRows, t.CandidateSource, boundary, firstLayer,
                         selection ? 1 : 0, candidateMask ? 1 : 0, candidateBefore ? 1 : 0,
                     })
                Fnv.Mix(ref h, (ulong)v);
            for (int il = 0; il < t.NLayer; il++)
            {
                Fnv.Mix(ref h, (ulong)(uint)t.Ratio[il]);
                Fnv.Mix(ref h, (ulong)(uint)t.KvSource[il]);
                Fnv.Mix(ref h, (ulong)(uint)t.IndexSource[il]);
                Fnv.Mix(ref h, (ulong)(uint)t.EngramIndex[il]);
                Fnv.Mix(ref h, (ulong)(uint)globalDevice[il]);
            }
            foreach (int s in kvGroups)
                Fnv.Mix(ref h, (ulong)(uint)s);
            foreach (int r in maskReaderRatios)
                Fnv.Mix(ref h, (ulong)(uint)r);
            LayoutHash = h;
        }

        /// <summary>What the boundary into global device <paramref name="boundary"/> carries.</summary>
        public static Dsv4BoundaryLayout Create(Dsv4Topology t, int[] globalDevice, int boundary)
        {
            int first = Array.IndexOf(globalDevice, boundary);
            if (first <= 0)
                throw new ArgumentException($"global device {boundary} holds no layer past a boundary", nameof(boundary));
            int[] last = t.GroupLastDevice(globalDevice);
            var kv = new List<int>();
            for (int s = 0; s < first; s++)
            {
                if (last[s] >= boundary)
                    kv.Add(s);
            }
            bool selection = false, candidateMask = false;
            var readers = new List<int>();
            for (int il = first; il < t.NLayer; il++)
            {
                // The layer attends over the selection of an index source before the boundary. A later device
                // reading one counts too: the device in between hands on what it received.
                if (t.V41 && t.Ratio[il] != 0 && t.IndexSource[il] >= 0 && t.IndexSource[il] < first)
                    selection = true;
                if (t.V41 && t.CandidateSource >= 0 && t.CandidateSource < first && t.Ratio[il] != 0
                    && t.IndexSource[il] == il && il > t.CandidateSource)
                {
                    candidateMask = true;
                    readers.Add(t.Ratio[il]);
                }
            }
            int c = t.CandidateSource;
            bool candidateBefore = t.V41 && c >= 0 && c < first && t.Ratio[c] != 0 && t.IndexSource[c] == c;
            return new Dsv4BoundaryLayout(t, boundary, first, kv.ToArray(), selection, candidateMask,
                candidateBefore, readers.ToArray(), globalDevice);
        }

        /// <summary>The cache blocks a ubatch at positions [p0, p0 + nt) completes in a group of
        /// <paramref name="ratio"/>: one per position p with (p + 1) % ratio == 0, numbered p / ratio.
        /// CompressV41's own arithmetic.</summary>
        public static (int FirstBlock, int Count) CommittedRange(long p0, int nt, int ratio)
        {
            if (ratio <= 0 || nt <= 0)
                return (0, 0);
            long firstBoundary = -1;
            for (long p = p0; p < p0 + nt; p++)
            {
                if ((p + 1) % ratio == 0)
                {
                    firstBoundary = p;
                    break;
                }
            }
            if (firstBoundary < 0)
                return (0, 0);
            int count = (int)((p0 + nt - 1 - firstBoundary) / ratio) + 1;
            return ((int)(firstBoundary / ratio), count);
        }

        /// <summary>The committed ranges a payload for a ubatch at [p0, p0 + nt) must carry, in
        /// <see cref="KvGroups"/> order; a group that completes no block in the ubatch has none.</summary>
        public Dsv4CommittedRange[] ExpectedRanges(long p0, int nt)
        {
            var list = new List<Dsv4CommittedRange>(KvGroups.Length);
            foreach (int s in KvGroups)
            {
                var (first, count) = CommittedRange(p0, nt, _t.Ratio[s]);
                if (count > 0)
                    list.Add(new Dsv4CommittedRange(s, first, count));
            }
            return list.ToArray();
        }

        /// <summary>Columns of the candidate mask the readers past the boundary read for this ubatch: each
        /// index source reads the rows visible at its ratio, (p + 1) / ratio for row t at p = p0 + t.</summary>
        public int MaskWidth(long p0, int nt)
        {
            long width = 0;
            foreach (int r in _maskReaderRatios)
                width = Math.Max(width, (p0 + nt) / r);
            return (int)Math.Min(width, _t.MaskRows);
        }

        /// <summary>Whether this ubatch's payload carries the mask.</summary>
        public bool CarriesMask(bool candidateActive) => CandidateMask && candidateActive;

        private static long Pad4(long v) => (v + 3) & ~3L;

        /// <summary>The body's sections, in wire order.</summary>
        public Dsv4Section[] Sections(int nt, ReadOnlySpan<Dsv4CommittedRange> ranges, bool candidateActive, int maskWidth)
        {
            var t = _t;
            var list = new List<Dsv4Section>(8 + 2 * ranges.Length);
            long off = 0;
            void Add(Dsv4SectionKind kind, long bytes, int range = -1, int rows = 0, int width = 0)
            {
                list.Add(new Dsv4Section(kind, off, bytes, range, rows, width));
                off += Pad4(bytes);
            }
            Add(Dsv4SectionKind.Ranges, 12L * ranges.Length);
            Add(Dsv4SectionKind.Xs, (long)nt * Hc * t.NEmbd * 4);
            if (t.V41)
                Add(Dsv4SectionKind.Gates, (long)nt * Hc * 4);
            for (int i = 0; i < ranges.Length; i++)
            {
                Add(Dsv4SectionKind.CompK, (long)ranges[i].Count * t.HeadDim * 2, i);
                Add(Dsv4SectionKind.LidK, (long)ranges[i].Count * t.IdxHeadSize * 2, i);
            }
            if (Selection)
            {
                Add(Dsv4SectionKind.TopkIdx, (long)nt * Math.Max(t.IdxTopK, 1) * 4);
                Add(Dsv4SectionKind.TopkCnt, (long)nt * 4);
            }
            if (CarriesMask(candidateActive) && maskWidth > 0)
                Add(Dsv4SectionKind.Mask, (long)nt * maskWidth, rows: nt, width: maskWidth);
            return list.ToArray();
        }

        /// <summary>The body's byte count (each section padded to 4 bytes).</summary>
        public static long BodyBytes(Dsv4Section[] sections)
        {
            if (sections.Length == 0)
                return 0;
            Dsv4Section last = sections[^1];
            return last.Offset + Pad4(last.Bytes);
        }

        /// <summary>The largest body a ubatch of this boundary can need: what the receiving staging is sized for.</summary>
        public long MaxBodyBytes()
        {
            var ranges = new Dsv4CommittedRange[KvGroups.Length];
            for (int i = 0; i < ranges.Length; i++)
                ranges[i] = new Dsv4CommittedRange(KvGroups[i], 0, _t.NUbatch / Math.Max(1, _t.Ratio[KvGroups[i]]) + 1);
            return BodyBytes(Sections(_t.NUbatch, ranges, candidateActive: true, maskWidth: _t.MaskRows));
        }

        public void WriteHeader(Span<int> words, long sequence, int p0, int nt, bool candidateActive,
            int rangeCount, int maskWidth, long bodyBytes)
        {
            words[0] = Magic;
            words[1] = Version;
            words[2] = (int)(uint)LayoutHash;
            words[3] = (int)(uint)(LayoutHash >> 32);
            words[4] = Boundary;
            words[5] = (int)(uint)sequence;
            words[6] = p0;
            words[7] = nt;
            words[8] = candidateActive ? 1 : 0;
            words[9] = rangeCount;
            words[10] = maskWidth;
            words[11] = checked((int)bodyBytes);
        }

        /// <summary>The header of the payload for this ubatch, checked against what this node expects;
        /// returns whether the candidate mask is active and the body's byte count. Throws on anything else,
        /// before the body is read.</summary>
        public (bool CandidateActive, int BodyBytes) ValidateHeader(ReadOnlySpan<int> words, long sequence, int p0, int nt)
        {
            void Check(bool ok, string what)
            {
                if (!ok)
                    throw new InvalidOperationException($"[dsv4-cuda] distributed boundary into device {Boundary}: {what}. " +
                        "The nodes disagree on the model, its placement or the position; every node must run the same " +
                        "checkpoint and settings, and a node that failed must be restarted with the others.");
            }
            Check(words.Length >= HeaderWords && words[0] == Magic, $"not a DSV4 boundary payload (magic 0x{(words.Length > 0 ? words[0] : 0):X8})");
            Check(words[1] == Version, $"payload version {words[1]}, this node speaks {Version}");
            ulong hash = (uint)words[2] | ((ulong)(uint)words[3] << 32);
            Check(hash == LayoutHash, $"layout hash 0x{hash:X16} differs from this node's 0x{LayoutHash:X16}");
            Check(words[4] == Boundary, $"payload for boundary {words[4]}");
            Check(words[5] == (int)(uint)sequence, $"boundary sequence {words[5]}, expected {(int)(uint)sequence}");
            Check(words[6] == p0 && words[7] == nt, $"payload for positions [{words[6]}, {words[6] + words[7]}), this ubatch is [{p0}, {p0 + nt})");
            bool active = words[8] != 0;
            Check(words[8] is 0 or 1 && active == CandidateBefore, $"candidate mask active={words[8]}, expected {(CandidateBefore ? 1 : 0)}");
            Dsv4CommittedRange[] expected = ExpectedRanges(p0, nt);
            Check(words[9] == expected.Length, $"{words[9]} committed ranges, expected {expected.Length}");
            int width = CarriesMask(active) ? MaskWidth(p0, nt) : 0;
            Check(words[10] == width, $"candidate mask width {words[10]}, expected {width}");
            long body = BodyBytes(Sections(nt, expected, active, width));
            Check(words[11] == body, $"body of {words[11]} bytes, expected {body}");
            return (active, (int)body);
        }

        /// <summary>The committed ranges as body words.</summary>
        public static void WriteRanges(Span<int> words, ReadOnlySpan<Dsv4CommittedRange> ranges)
        {
            for (int i = 0; i < ranges.Length; i++)
            {
                words[3 * i] = ranges[i].Source;
                words[3 * i + 1] = ranges[i].FirstBlock;
                words[3 * i + 2] = ranges[i].Count;
            }
        }

        /// <summary>The ranges a body carries, each equal to the one recomputed from (p0, nt, ratio). Throws
        /// on the first that differs, before any of the body reaches the device.</summary>
        public Dsv4CommittedRange[] ValidateRanges(ReadOnlySpan<int> words, int p0, int nt)
        {
            Dsv4CommittedRange[] expected = ExpectedRanges(p0, nt);
            for (int i = 0; i < expected.Length; i++)
            {
                var got = new Dsv4CommittedRange(words[3 * i], words[3 * i + 1], words[3 * i + 2]);
                if (got != expected[i])
                    throw new InvalidOperationException(
                        $"[dsv4-cuda] distributed boundary into device {Boundary}: committed range {i} is (source {got.Source}, " +
                        $"blocks {got.FirstBlock}+{got.Count}), but positions [{p0}, {p0 + nt}) commit (source " +
                        $"{expected[i].Source}, blocks {expected[i].FirstBlock}+{expected[i].Count}). The sending node's " +
                        "caches are not the ones this node's layers expect.");
            }
            return expected;
        }

        /// <summary>
        /// The check a LOCAL boundary (two GPUs of one process) makes before it hands its state over, the
        /// counterpart of the network receiver's <see cref="ValidateRanges"/>: for every group the layers past the
        /// boundary read, the committed ranges noted in this ubatch, in the order noted, must be exactly the blocks
        /// the ubatch closes. That is one range of the active slot for a ubatch at [p0, p0 + nt), or one block per
        /// closing row of a batched decode step (row i: <paramref name="rowSlots"/>[i] at
        /// <paramref name="rowPositions"/>[i]). The candidate mask must be active when the layers past the boundary
        /// prune with it. A stage that RECEIVED a group's rows notes them as committed, so its local boundaries
        /// and the next stage hand them on; a stage that lost them would otherwise let the next device attend
        /// over the replica's zeroed rows without any error. Throws on the first difference.
        /// </summary>
        public void ValidateHandOver<TSlot>(List<(int Source, TSlot Slot, int FirstBlock, int Count)> committed,
            TSlot active, long p0, int nt, TSlot[] rowSlots, int[] rowPositions, bool candidateActive) where TSlot : class
        {
            foreach (int source in KvGroups)
            {
                int ratio = _t.Ratio[source];
                int at = 0;
                void Expect(TSlot slot, long first, int width)
                {
                    var (block, count) = CommittedRange(first, width, ratio);
                    if (count == 0)
                        return;
                    int i = NextOf(committed, source, ref at);
                    if (i < 0 || !ReferenceEquals(committed[i].Slot, slot) || committed[i].FirstBlock != block
                        || committed[i].Count != count)
                        throw HandOverMismatch(source, committed, first, width,
                            i < 0 ? "no committed range" : $"blocks {committed[i].FirstBlock}+{committed[i].Count}" +
                                (ReferenceEquals(committed[i].Slot, slot) ? "" : " of another sequence"),
                            $"blocks {block}+{count}");
                    at = i + 1;
                }
                if (rowSlots == null)
                    Expect(active, p0, nt);
                else
                {
                    for (int r = 0; r < rowSlots.Length; r++)
                        Expect(rowSlots[r], rowPositions[r], 1);
                }
                int extra = NextOf(committed, source, ref at);
                if (extra >= 0)
                    throw HandOverMismatch(source, committed, p0, nt,
                        $"an extra range (blocks {committed[extra].FirstBlock}+{committed[extra].Count})", "nothing more");
            }
            if (CandidateMask && candidateActive != CandidateBefore)
                throw new InvalidOperationException(
                    $"[dsv4-cuda] boundary into device {Boundary}: the candidate mask is {(candidateActive ? "" : "not ")}active, but " +
                    $"layer {_t.CandidateSource} {(CandidateBefore ? "ran" : "did not run")} before it; the layers past the boundary " +
                    "would prune with a stale mask.");
        }

        private static int NextOf<TSlot>(List<(int Source, TSlot Slot, int FirstBlock, int Count)> committed, int source, ref int at)
        {
            for (int i = at; i < committed.Count; i++)
            {
                if (committed[i].Source == source)
                {
                    at = i;
                    return i;
                }
            }
            at = committed.Count;
            return -1;
        }

        private InvalidOperationException HandOverMismatch<TSlot>(int source,
            List<(int Source, TSlot Slot, int FirstBlock, int Count)> committed, long p0, int nt, string got, string expected)
            => new InvalidOperationException(
                $"[dsv4-cuda] boundary into device {Boundary}: group {source} would hand over {got} for positions " +
                $"[{p0}, {p0 + nt}), which commit {expected} ({committed.Count} committed range(s) noted in all). The " +
                "layers past this boundary read those rows from their replica; a stage that received them must hand " +
                "them on as if its own layer had committed them.");
    }

    /// <summary>What crossed the boundaries of one engine, for tests and TS_DSV4_PERF: the network ones
    /// (sent and received payloads) and the local ones (what each in-process hand-over carried besides the
    /// hidden streams).</summary>
    internal sealed class Dsv4BoundaryStats
    {
        public long Sent, Received, BytesSent, BytesReceived;
        public long CommittedRowsSent, CommittedRowsReceived, SelectionsSent, SelectionsReceived;
        public long MasksSent, MasksReceived, MaskBytesSent, MaskBytesReceived;
        /// <summary>Local boundaries: hand-overs, committed rows, selections and candidate masks they carried.</summary>
        public long LocalHandOvers, LocalRowsHandedOver, LocalSelectionsHandedOver, LocalMasksHandedOver;
        /// <summary>Local hand-overs per source group.</summary>
        public readonly Dictionary<int, long> LocalRowsBySource = new Dictionary<int, long>();
        /// <summary>Wall time of the boundaries: DtoH + stream sync + network on the sender, network +
        /// validation + queueing the HtoD on the receiver.</summary>
        public long SendTicks, ReceiveTicks;
        /// <summary>Engram rows that crossed between an owning and a reading node: messages, bytes (header and body)
        /// and wall time (gather + send on the owner, receive on the reader).</summary>
        public long EngramMessagesSent, EngramMessagesReceived, EngramBytesSent, EngramBytesReceived, EngramTicks;
        /// <summary>Deferred sends (TS_DSV4_ASYNC_SEND): how many payloads left one step late, and the host time spent
        /// waiting for their device-to-host copies when they did. The receiver's time waiting for its one staging to
        /// be copied out before the next payload may land there (non-zero in steady state means the single receive
        /// slot is what holds the sender back).</summary>
        public long Deferred, SendReadyWaitTicks, ReceiveSlotWaitTicks;
        public readonly HashSet<int> SourcesSent = new HashSet<int>(), SourcesReceived = new HashSet<int>();
    }
}
