// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Globalization;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>One node's part of a <see cref="ModelBase.EvictHostTablePageCache"/> call.</summary>
public sealed class HostTableEvictionNode
{
    public int Node { get; init; }
    public bool Supported { get; init; }
    public int Ranges { get; init; }
    public double BytesMiB { get; init; }
    /// <summary>Page-cache MiB of the tables before / after the eviction; negative when not measured.</summary>
    public double ResidentBeforeMiB { get; init; }
    public double ResidentAfterMiB { get; init; }
    /// <summary>See <see cref="HostPageCacheReport.Verified"/>.</summary>
    public bool? Verified { get; init; }
    public int MappingsAdvised { get; init; }
    public int MappingsSkipped { get; init; }
    public int Errors { get; init; }
    public double ElapsedMs { get; init; }
    public double EvictMs { get; init; }

    internal const int Fields = 12;

    internal static HostTableEvictionNode From(int node, HostPageCacheReport r) => new()
    {
        Node = node,
        Supported = r.Supported,
        Ranges = r.Ranges.Count,
        BytesMiB = r.Bytes / 1048576.0,
        ResidentBeforeMiB = r.ResidentBefore < 0 ? -1 : r.ResidentBefore / 1048576.0,
        ResidentAfterMiB = r.ResidentAfter < 0 ? -1 : r.ResidentAfter / 1048576.0,
        Verified = r.Verified,
        MappingsAdvised = r.MappingsAdvised,
        MappingsSkipped = r.MappingsSkipped,
        Errors = r.Errors,
        ElapsedMs = r.ElapsedMs,
        EvictMs = r.EvictMs,
    };

    /// <summary>This node's slot of the cross-node exchange: every other slot stays zero, so a sum gathers them.</summary>
    internal void WriteSlot(float[] slots)
    {
        int at = Node * Fields;
        slots[at] = 1;
        slots[at + 1] = Supported ? 1 : 0;
        slots[at + 2] = Ranges;
        slots[at + 3] = (float)BytesMiB;
        slots[at + 4] = (float)ResidentBeforeMiB;
        slots[at + 5] = (float)ResidentAfterMiB;
        slots[at + 6] = Verified is bool v ? (v ? 1 : 0) : -1;
        slots[at + 7] = MappingsAdvised;
        slots[at + 8] = MappingsSkipped;
        slots[at + 9] = Errors;
        slots[at + 10] = (float)ElapsedMs;
        slots[at + 11] = (float)EvictMs;
    }

    /// <summary>Node <paramref name="node"/>'s slot, or null when that node wrote none.</summary>
    internal static HostTableEvictionNode ReadSlot(float[] slots, int node)
    {
        int at = node * Fields;
        if (slots[at] != 1)
            return null;
        return new HostTableEvictionNode
        {
            Node = node,
            Supported = slots[at + 1] == 1,
            Ranges = (int)slots[at + 2],
            BytesMiB = slots[at + 3],
            ResidentBeforeMiB = slots[at + 4],
            ResidentAfterMiB = slots[at + 5],
            Verified = slots[at + 6] switch { 1 => true, 0 => false, _ => null },
            MappingsAdvised = (int)slots[at + 7],
            MappingsSkipped = (int)slots[at + 8],
            Errors = (int)slots[at + 9],
            ElapsedMs = slots[at + 10],
            EvictMs = slots[at + 11],
        };
    }

    /// <summary>key=value fields for a log line.</summary>
    public string Describe()
    {
        var c = CultureInfo.InvariantCulture;
        string Mib(double v) => v < 0 ? "na" : v.ToString("F1", c);
        string verified = Verified is bool v ? (v ? "true" : "false") : "na";
        return string.Format(c,
            "node={0} supported={1} ranges={2} bytesMiB={3} residentBeforeMiB={4} residentAfterMiB={5} verified={6} " +
            "advised={7} skipped={8} errors={9} elapsedMs={10:F0} evictMs={11:F0}",
            Node, Supported ? "true" : "false", Ranges, Mib(BytesMiB), Mib(ResidentBeforeMiB), Mib(ResidentAfterMiB),
            verified, MappingsAdvised, MappingsSkipped, Errors, ElapsedMs, EvictMs);
    }
}

/// <summary>The outcome of <see cref="ModelBase.EvictHostTablePageCache"/> on every node.</summary>
public sealed class HostTableEvictionResult
{
    /// <summary>This process's full report.</summary>
    public HostPageCacheReport Local { get; init; }

    /// <summary>One entry per node, node 0 first (just this node in a single process). A node whose report could
    /// not be gathered (a transport without a cross-node reduction) is missing.</summary>
    public IReadOnlyList<HostTableEvictionNode> Nodes { get; init; } = Array.Empty<HostTableEvictionNode>();

    /// <summary>The node count of the group, so a missing report is visible.</summary>
    public int NodeCount { get; init; } = 1;

    /// <summary>True only when every node reported, could evict, and measured its tables out of the page cache;
    /// false when any node measured them still cached or could not evict; null when nothing decides it (no
    /// measurement, or a node's report is missing).</summary>
    public bool? Verified
    {
        get
        {
            if (Nodes.Count < NodeCount)
                return null;
            bool unknown = false;
            foreach (var n in Nodes)
            {
                if (!n.Supported || n.Verified == false) return false;
                if (n.Verified == null) unknown = true;
            }
            return unknown ? null : true;
        }
    }
}

public abstract partial class ModelBase
{
    private const int TpControlEvictHostTables = 6;

    /// <summary>
    /// The host-mapped tables of this checkpoint whose page cache decides how fast a request runs: tables read a
    /// few scattered rows per token straight from the model file (DeepSeek V4.1's Engram tables). A benchmark drops
    /// them from the page cache with <see cref="EvictHostTablePageCache"/> to measure a cold row inside one process.
    /// Empty for a model that reads no such table. Depends on the checkpoint only, not on which node holds which
    /// layer, so every node of a distributed group evicts the same ranges.
    /// </summary>
    public virtual IReadOnlyList<HostFileRange> HostMappedTables => Array.Empty<HostFileRange>();

    /// <summary>
    /// Drop <see cref="HostMappedTables"/> from the page cache on every node, without unloading the model: the
    /// driver tells the workers first (the worker loop runs the same eviction), each node evicts its own page cache,
    /// and the nodes then exchange their reports, which doubles as the barrier, so no node starts the next forward
    /// before every node has evicted. Linux only; elsewhere every node reports <c>supported=false</c>.
    /// </summary>
    public HostTableEvictionResult EvictHostTablePageCache(HostPageCacheMeasure measure)
    {
        ThrowIfBackendFailed();
        // A multi-node eviction ends in a cross-node exchange: started anywhere but on the driver, it would wait for
        // nodes that were never told to join it.
        if (DistributedControlGroup?.NodeCount > 1 && !_distributedDriver)
            throw new InvalidOperationException("Only the distributed driver (after BeginDistributedDriver) starts a page-cache eviction.");
        if (_distributedDriver)
            DistributedControlGroup.BroadcastControl(TpControlEvictHostTables, new[] { (int)measure });
        try
        {
            return EvictHostTablesAndGather(measure);
        }
        catch (Exception failure)
        {
            AbortDistributedExecution(failure);
            throw;
        }
    }

    private HostTableEvictionResult EvictHostTablesAndGather(HostPageCacheMeasure measure)
    {
        if (!Enum.IsDefined(measure))
            throw new ArgumentOutOfRangeException(nameof(measure), $"Unknown page-cache measurement {(int)measure}.");
        // Nothing of this process may read the tables back while (or right after) they are dropped.
        QuiesceHostTableReaders();
        HostPageCacheReport local = HostPageCache.Evict(HostMappedTables, measure);
        ITensorParallelGroup group = DistributedControlGroup;
        int nodes = group?.NodeCount > 1 ? group.NodeCount : 1;
        int node = nodes > 1 ? HostTableEvictionNodeIndex(group) : 0;
        var mine = HostTableEvictionNode.From(node, local);
        Console.WriteLine("host-table eviction: " + mine.Describe());

        var all = new List<HostTableEvictionNode>();
        if (nodes == 1)
        {
            all.Add(mine);
        }
        else if (group is INestedTensorParallelGroup nested)
        {
            var slots = new float[HostTableEvictionNode.Fields * nodes];
            mine.WriteSlot(slots);
            nested.CrossNodeAllReduce(slots, slots.Length);
            for (int k = 0; k < nodes; k++)
                if (HostTableEvictionNode.ReadSlot(slots, k) is { } entry)
                    all.Add(entry);
        }
        else
        {
            // No cross-node reduction to carry the reports: still wait for every node.
            group.Barrier();
            all.Add(mine);
        }
        return new HostTableEvictionResult { Local = local, Nodes = all, NodeCount = nodes };
    }

    private static int HostTableEvictionNodeIndex(ITensorParallelGroup group)
        => group is ILayerSplitGroup layers ? layers.NodeId
            : group.Degree > 0 ? group.GlobalRankOffset / group.Degree : 0;

    /// <summary>
    /// Before <see cref="EvictHostTablePageCache"/> drops <see cref="HostMappedTables"/> on this node: stop whatever this
    /// model reads from them on its own between forwards (a read-ahead, a background warm), and return once it has
    /// stopped, so the eviction's "after" measurement and the next forward see the tables cold. No-op by default.
    /// </summary>
    protected virtual void QuiesceHostTableReaders()
    {
    }

    /// <summary>The worker side of <see cref="EvictHostTablePageCache"/>.</summary>
    private void RunHostTableEvictionControl(int[] payload)
        => EvictHostTablesAndGather(payload.Length > 0 ? (HostPageCacheMeasure)payload[0] : HostPageCacheMeasure.None);
}
