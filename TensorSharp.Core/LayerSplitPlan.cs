using System;

namespace TensorSharp;

/// <summary>Deterministic contiguous whole-layer ownership, independent of model family.</summary>
public sealed class LayerSplitPlan
{
    public int LayerCount { get; }
    public int LocalDegree { get; }
    public int NodeId { get; }
    public int NodeCount { get; }
    /// <summary>Whole-layer stages over every node (LocalDegree x NodeCount unless the nodes' shapes differ).</summary>
    public int GlobalDegree { get; }
    public int FirstLayer { get; }
    public int EndLayer { get; }
    private readonly int[] _localBoundaries;

    /// <summary>This node's position in the pipeline. It equals <see cref="NodeId"/>
    /// unless <see cref="DriverRunsLastStage"/> rotates the order.</summary>
    public int StageIndex { get; }

    /// <summary>Node 0 (the driver, which samples) runs the last stage, and node 1
    /// the first. The output projection then runs where its logits are consumed,
    /// so a full-vocabulary vector never crosses the network; workers receive the
    /// tokens with each forward control and can embed them.</summary>
    public bool DriverRunsLastStage { get; }

    public bool IsFirst => StageIndex == 0;
    public bool IsLast => StageIndex == NodeCount - 1;
    public int PreviousNode => NodeOfStage(StageIndex - 1);
    public int NextNode => NodeOfStage(StageIndex + 1);

    /// <summary>Environment variable choosing which pipeline stage node 0 runs: <c>last</c> (default) or <c>first</c>.</summary>
    public const string DriverStageVariable = "TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE";

    public LayerSplitPlan(int layerCount, int localDegree, int nodeId = 0, int nodeCount = 1)
        : this(layerCount, localDegree, nodeId, nodeCount, false)
    {
    }

    public LayerSplitPlan(int layerCount, int localDegree, int nodeId, int nodeCount, bool driverRunsLastStage)
    {
        if (layerCount < 1) throw new ArgumentOutOfRangeException(nameof(layerCount));
        if (localDegree < 1) throw new ArgumentOutOfRangeException(nameof(localDegree));
        if (nodeCount < 1) throw new ArgumentOutOfRangeException(nameof(nodeCount));
        if ((uint)nodeId >= (uint)nodeCount) throw new ArgumentOutOfRangeException(nameof(nodeId));
        if (layerCount < checked(localDegree * nodeCount))
            throw new NotSupportedException($"Cannot distribute {layerCount} layers over {localDegree * nodeCount} GPUs: each GPU must own at least one layer.");
        LayerCount = layerCount;
        LocalDegree = localDegree;
        NodeId = nodeId;
        NodeCount = nodeCount;
        GlobalDegree = checked(localDegree * nodeCount);
        DriverRunsLastStage = driverRunsLastStage && nodeCount > 1;
        StageIndex = StageOf(nodeId, nodeCount, DriverRunsLastStage);
        FirstLayer = (int)((long)LayerCount * StageIndex / NodeCount);
        EndLayer = (int)((long)LayerCount * (StageIndex + 1) / NodeCount);
        _localBoundaries = new int[localDegree + 1];
        for (int rank = 0; rank <= localDegree; rank++)
            _localBoundaries[rank] = (int)((long)layerCount * (StageIndex * localDegree + rank) / GlobalDegree);
    }

    private LayerSplitPlan(int layerCount, int localDegree, int nodeId, int nodeCount, bool driverRunsLastStage,
        int[] stageBoundaries, int[] localBoundaries) : this(layerCount, localDegree, nodeId, nodeCount, driverRunsLastStage)
    {
        FirstLayer = stageBoundaries[StageIndex];
        EndLayer = stageBoundaries[StageIndex + 1];
        _localBoundaries = localBoundaries;
    }

    /// <summary>A node of a pipeline whose nodes differ (<see cref="LayerSplitTopology"/>): every field given.</summary>
    private LayerSplitPlan(int layerCount, int localDegree, int nodeId, int nodeCount, int globalDegree,
        bool driverRunsLastStage, int stageIndex, int firstLayer, int endLayer, int[] localBoundaries)
    {
        LayerCount = layerCount;
        LocalDegree = localDegree;
        NodeId = nodeId;
        NodeCount = nodeCount;
        GlobalDegree = globalDegree;
        DriverRunsLastStage = driverRunsLastStage;
        StageIndex = stageIndex;
        FirstLayer = firstLayer;
        EndLayer = endLayer;
        _localBoundaries = localBoundaries;
    }

    /// <summary>
    /// Node <paramref name="nodeId"/>'s share of a pipeline over <paramref name="topology"/>'s nodes. A uniform
    /// topology gives exactly the plan of <see cref="CreateWeighted(long[], int, int, int, long, long, bool)"/>.
    /// Otherwise each node's layers are proportional to its capacity (its stages x their capacity): the maximum of a
    /// node's resident bytes over its capacity is minimized, every stage owning at least one layer; within the node,
    /// the layers split over its equal stages as in the uniform plan. Every node computes the same boundaries from the
    /// same agreed topology.
    /// </summary>
    /// <param name="layerBytes">Resident bytes per layer; null weighs every layer alike.</param>
    public static LayerSplitPlan Create(int layerCount, long[] layerBytes, LayerSplitTopology topology, int nodeId,
        long embeddingBytes, long headBytes, bool driverRunsLastStage)
    {
        ArgumentNullException.ThrowIfNull(topology);
        if (layerBytes != null && layerBytes.Length != layerCount)
            throw new ArgumentException("One byte count per layer is required.", nameof(layerBytes));
        int nodeCount = topology.NodeCount;
        if ((uint)nodeId >= (uint)nodeCount) throw new ArgumentOutOfRangeException(nameof(nodeId));
        int localStages = topology.Stages(nodeId);
        if (topology.IsUniform)
            return layerBytes == null
                ? new LayerSplitPlan(layerCount, localStages, nodeId, nodeCount, driverRunsLastStage)
                : CreateWeighted(layerBytes, localStages, nodeId, nodeCount, embeddingBytes, headBytes, driverRunsLastStage);
        if (layerCount < topology.TotalStages)
            throw new NotSupportedException(
                $"Cannot distribute {layerCount} layers over {topology.TotalStages} stages: each stage must own at least one layer.");
        bool driverLast = driverRunsLastStage && nodeCount > 1;
        long[] bytes = layerBytes ?? Ones(layerCount);
        if (layerBytes == null) embeddingBytes = headBytes = 0;
        // Pipeline position p is run by node NodeOfStage(p).
        var capacity = new long[nodeCount];
        var minimum = new int[nodeCount];
        for (int p = 0; p < nodeCount; p++)
        {
            int node = NodeOfStage(p, nodeCount, driverLast);
            capacity[p] = checked(topology.Stages(node) * topology.StageCapacity(node));
            minimum[p] = topology.Stages(node);
        }
        int[] boundaries = PartitionByCapacity(bytes, capacity, minimum, embeddingBytes, headBytes);
        int stage = StageOf(nodeId, nodeCount, driverLast);
        int first = boundaries[stage], end = boundaries[stage + 1];
        long[] owned = bytes.AsSpan(first, end - first).ToArray();
        int[] local = Partition(owned, localStages, 1,
            stage == 0 ? embeddingBytes : 0, stage == nodeCount - 1 ? headBytes : 0);
        for (int rank = 0; rank < local.Length; rank++) local[rank] += first;
        return new LayerSplitPlan(layerCount, localStages, nodeId, nodeCount, topology.TotalStages, driverLast,
            stage, first, end, local);
    }

    private static long[] Ones(int count)
    {
        var ones = new long[count];
        Array.Fill(ones, 1L);
        return ones;
    }

    /// <summary>
    /// Exact partition of an ordered layer sequence into parts of different capacities, minimizing the largest
    /// bytes-to-capacity ratio (compared exactly, by cross-multiplication). Part p owns at least
    /// <paramref name="minimumLayers"/>[p] layers. Equal-cost solutions choose the smallest preceding boundary.
    /// </summary>
    public static int[] PartitionByCapacity(long[] layerBytes, long[] capacity, int[] minimumLayers,
        long firstBytes = 0, long lastBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(layerBytes);
        ArgumentNullException.ThrowIfNull(capacity);
        ArgumentNullException.ThrowIfNull(minimumLayers);
        int partitions = capacity.Length;
        if (partitions < 1 || minimumLayers.Length != partitions)
            throw new ArgumentException("Every partition needs a capacity and a minimum layer count.");
        if (firstBytes < 0 || lastBytes < 0) throw new ArgumentOutOfRangeException(nameof(firstBytes));
        int count = layerBytes.Length;
        var before = new int[partitions + 1];   // layers the parts before p need at least
        for (int p = 0; p < partitions; p++)
        {
            if (capacity[p] < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
            if (minimumLayers[p] < 1) throw new ArgumentOutOfRangeException(nameof(minimumLayers));
            before[p + 1] = checked(before[p] + minimumLayers[p]);
        }
        if (before[partitions] > count)
            throw new ArgumentException("Every partition must own its minimum number of layers.");
        long[] prefix = new long[count + 1];
        for (int layer = 0; layer < count; layer++)
        {
            if (layerBytes[layer] < 0) throw new ArgumentOutOfRangeException(nameof(layerBytes));
            prefix[layer + 1] = checked(prefix[layer] + layerBytes[layer]);
        }
        _ = checked(prefix[count] + firstBytes + lastBytes);
        // best[p, end]: the smallest achievable maximum ratio, as (bytes, capacity), of parts 1..p ending at end.
        var bestBytes = new long[partitions + 1, count + 1];
        var bestCapacity = new long[partitions + 1, count + 1];
        var cuts = new int[partitions + 1, count + 1];
        var reachable = new bool[partitions + 1, count + 1];
        reachable[0, 0] = true;
        bestCapacity[0, 0] = 1;
        for (int p = 1; p <= partitions; p++)
        {
            int after = before[partitions] - before[p];   // layers the parts after p need at least
            for (int end = before[p]; end <= count - after; end++)
            for (int begin = before[p - 1]; begin <= end - minimumLayers[p - 1]; begin++)
            {
                if (!reachable[p - 1, begin]) continue;
                long bytes = prefix[end] - prefix[begin] + (p == 1 ? firstBytes : 0) + (p == partitions ? lastBytes : 0);
                // max(previous, bytes / capacity[p-1]) as a fraction.
                long maxBytes = bestBytes[p - 1, begin], maxCapacity = bestCapacity[p - 1, begin];
                if (Less(maxBytes, maxCapacity, bytes, capacity[p - 1]))
                {
                    maxBytes = bytes;
                    maxCapacity = capacity[p - 1];
                }
                if (!reachable[p, end] || Less(maxBytes, maxCapacity, bestBytes[p, end], bestCapacity[p, end]))
                {
                    reachable[p, end] = true;
                    bestBytes[p, end] = maxBytes;
                    bestCapacity[p, end] = maxCapacity;
                    cuts[p, end] = begin;
                }
            }
        }
        int[] boundaries = new int[partitions + 1];
        boundaries[partitions] = count;
        for (int p = partitions; p > 0; p--)
            boundaries[p - 1] = cuts[p, boundaries[p]];
        return boundaries;

        static bool Less(long aBytes, long aCapacity, long bBytes, long bCapacity)
            => (Int128)aBytes * bCapacity < (Int128)bBytes * aCapacity;
    }

    /// <summary>Pipeline position of a node.</summary>
    public static int StageOf(int nodeId, int nodeCount, bool driverRunsLastStage)
    {
        if (nodeCount < 1 || (uint)nodeId >= (uint)nodeCount) throw new ArgumentOutOfRangeException(nameof(nodeId));
        return driverRunsLastStage ? (nodeId + nodeCount - 1) % nodeCount : nodeId;
    }

    /// <summary>Node that runs a pipeline position.</summary>
    public static int NodeOfStage(int stage, int nodeCount, bool driverRunsLastStage)
    {
        if (nodeCount < 1 || (uint)stage >= (uint)nodeCount) throw new ArgumentOutOfRangeException(nameof(stage));
        return driverRunsLastStage ? (stage + 1) % nodeCount : stage;
    }

    public int NodeOfStage(int stage) => NodeOfStage(stage, NodeCount, DriverRunsLastStage);

    /// <summary>Reads <see cref="DriverStageVariable"/>. Every node must agree; adapters
    /// include the result in their startup configuration identity.</summary>
    public static bool DriverRunsLastStageFromEnvironment()
    {
        string value = Environment.GetEnvironmentVariable(DriverStageVariable);
        if (string.IsNullOrWhiteSpace(value)) return true;
        return value.Trim().ToLowerInvariant() switch
        {
            "last" => true,
            "first" => false,
            _ => throw new ArgumentException($"{DriverStageVariable} must be 'last' or 'first'."),
        };
    }

    /// <summary>Minimize the maximum estimated resident bytes per node, then per
    /// local device, including the embedding and head at their actual endpoints.</summary>
    public static LayerSplitPlan CreateWeighted(long[] layerBytes, int localDegree,
        int nodeId = 0, int nodeCount = 1, long embeddingBytes = 0, long headBytes = 0)
        => CreateWeighted(layerBytes, localDegree, nodeId, nodeCount, embeddingBytes, headBytes, false);

    public static LayerSplitPlan CreateWeighted(long[] layerBytes, int localDegree,
        int nodeId, int nodeCount, long embeddingBytes, long headBytes, bool driverRunsLastStage)
    {
        ArgumentNullException.ThrowIfNull(layerBytes);
        // Validate topology before indexing or allocating the dynamic program.
        var topology = new LayerSplitPlan(layerBytes.Length, localDegree, nodeId, nodeCount, driverRunsLastStage);
        int stage = topology.StageIndex;
        int[] stages = Partition(layerBytes, nodeCount, localDegree, embeddingBytes, headBytes);
        int first = stages[stage], end = stages[stage + 1];
        long[] owned = layerBytes.AsSpan(first, end - first).ToArray();
        int[] devices = Partition(owned, localDegree, 1,
            stage == 0 ? embeddingBytes : 0, stage == nodeCount - 1 ? headBytes : 0);
        for (int rank = 0; rank < devices.Length; rank++) devices[rank] += first;
        return new LayerSplitPlan(layerBytes.Length, localDegree, nodeId, nodeCount, topology.DriverRunsLastStage, stages, devices);
    }

    /// <summary>Exact minimax partition of an ordered layer sequence. Equal-cost
    /// solutions choose the smallest preceding boundary, deterministically.</summary>
    public static int[] Partition(long[] layerBytes, int partitions, int minimumLayers,
        long firstBytes = 0, long lastBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(layerBytes);
        if (partitions < 1 || minimumLayers < 1 ||
            (long)partitions * minimumLayers > layerBytes.Length)
            throw new ArgumentException("Every partition must own its minimum number of layers.");
        if (firstBytes < 0 || lastBytes < 0) throw new ArgumentOutOfRangeException(nameof(firstBytes));
        int count = layerBytes.Length;
        long[] prefix = new long[count + 1];
        for (int layer = 0; layer < count; layer++)
        {
            if (layerBytes[layer] < 0) throw new ArgumentOutOfRangeException(nameof(layerBytes));
            prefix[layer + 1] = checked(prefix[layer] + layerBytes[layer]);
        }
        _ = checked(prefix[count] + firstBytes + lastBytes);
        var best = new long[partitions + 1, count + 1];
        var cuts = new int[partitions + 1, count + 1];
        var reachable = new bool[partitions + 1, count + 1];
        reachable[0, 0] = true;
        for (int stage = 1; stage <= partitions; stage++)
        for (int end = stage * minimumLayers; end <= count - (partitions - stage) * minimumLayers; end++)
        for (int begin = (stage - 1) * minimumLayers; begin <= end - minimumLayers; begin++)
        {
            if (!reachable[stage - 1, begin]) continue;
            long bytes = prefix[end] - prefix[begin] + (stage == 1 ? firstBytes : 0) +
                (stage == partitions ? lastBytes : 0);
            long maximum = Math.Max(best[stage - 1, begin], bytes);
            if (!reachable[stage, end] || maximum < best[stage, end])
            {
                reachable[stage, end] = true;
                best[stage, end] = maximum;
                cuts[stage, end] = begin;
            }
        }
        int[] boundaries = new int[partitions + 1];
        boundaries[partitions] = count;
        for (int stage = partitions; stage > 0; stage--)
            boundaries[stage - 1] = cuts[stage, boundaries[stage]];
        return boundaries;
    }

    public bool Owns(int layer) => layer >= FirstLayer && layer < EndLayer;
    public int LocalRank(int layer)
    {
        if (!Owns(layer)) throw new ArgumentOutOfRangeException(nameof(layer), "Layer belongs to a different node.");
        int rank = 0;
        while (layer >= _localBoundaries[rank + 1]) rank++;
        return rank;
    }

    public static bool TryGetLayer(string weightName, out int layer)
    {
        layer = -1;
        if (weightName == null || !weightName.StartsWith("blk.", StringComparison.Ordinal)) return false;
        int separator = weightName.IndexOf('.', 4);
        return separator > 4 && int.TryParse(weightName.AsSpan(4, separator - 4), out layer) && layer >= 0;
    }
}
