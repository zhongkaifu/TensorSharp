using System;
using System.Net;

namespace TensorSharp.Distributed;

/// <summary>
/// Whole-layer node pipeline using the same peer mesh and driver protocol as TP. Nodes may differ: each runs its own
/// number of whole-layer stages, each stage a tensor-parallel group of its own size, and the nodes exchange their
/// shapes and capacities at startup (<see cref="Topology"/>) so every node derives the same layer plan.
/// </summary>
public sealed class DistributedLayerSplitGroup : ILayerSplitGroup, INestedTensorParallelGroup, IDistributedExecutionGroup
{
    private readonly TcpCommunicator _tcp;
    private bool _disposed;
    public ITensorParallelGroup LocalGroup { get; }
    public int NodeId { get; }
    public int Degree => LocalGroup.Degree;
    public int NodeCount { get; }
    public int StageTensorParallel { get; }
    public int LocalStages => Degree / StageTensorParallel;
    public LayerSplitTopology Topology { get; }
    public int GlobalDegree => Topology.TotalDevices;
    public int GlobalRankOffset => Topology.DeviceOffset(NodeId);
    // Layer placement must never activate the tensor sharding paths.
    public bool IsActive => false;

    /// <param name="localGroup">This node's devices: P x T of them, stage s on devices [s*T, (s+1)*T).</param>
    /// <param name="stageTensorParallel">T, the devices of each local stage.</param>
    /// <param name="stageCapacity">One local stage's capacity (its devices' memory in whole GiB); 0 when unknown,
    /// which weighs every device alike on every node.</param>
    public DistributedLayerSplitGroup(ITensorParallelGroup localGroup, int nodeId, IPEndPoint[] endpoints,
        int stageTensorParallel = 1, long stageCapacity = 0)
    {
        ArgumentNullException.ThrowIfNull(localGroup);
        var config = new DistributedTpConfig(nodeId, localGroup.Degree, endpoints);
        if (stageTensorParallel < 1 || localGroup.Degree % stageTensorParallel != 0)
        {
            localGroup.Dispose();
            throw new ArgumentException($"{localGroup.Degree} local devices do not form stages of {stageTensorParallel}.", nameof(stageTensorParallel));
        }
        if (stageCapacity < 0 || stageCapacity >= 1L << 24)
        {
            localGroup.Dispose();
            throw new ArgumentOutOfRangeException(nameof(stageCapacity));
        }
        LocalGroup = localGroup;
        NodeId = nodeId;
        NodeCount = config.PeerEndpoints.Length;
        StageTensorParallel = stageTensorParallel;
        try
        {
            _tcp = new TcpCommunicator(nodeId, config.PeerEndpoints);
            // Reject a node in another parallel mode before any model weight is loaded; the nodes' shapes may differ.
            _tcp.ValidateConfiguration("layer-split");
            Topology = ExchangeShapes(stageCapacity);
            Console.WriteLine($"Distributed layer splitting: node {NodeId}/{NodeCount}, {LocalStages} local stage(s) x " +
                $"{StageTensorParallel} GPU(s), {GlobalDegree} GPUs in total ({Topology.Describe()}).");
        }
        catch
        {
            _tcp?.Dispose();
            LocalGroup.Dispose();
            throw;
        }
    }

    /// <summary>Every node's (stages, tensor-parallel degree, stage capacity), summed into place: each node fills its
    /// own row. The values are small integers, exact in a float. A node that cannot measure its devices reports 0,
    /// and then every node weighs a stage by its device count instead.</summary>
    private LayerSplitTopology ExchangeShapes(long stageCapacity)
    {
        var rows = new float[3 * NodeCount];
        rows[3 * NodeId] = LocalStages;
        rows[3 * NodeId + 1] = StageTensorParallel;
        rows[3 * NodeId + 2] = stageCapacity;
        _tcp.AllReduce(rows, rows.Length);
        var stages = new int[NodeCount];
        var tensorParallel = new int[NodeCount];
        var capacity = new long[NodeCount];
        bool measured = true;
        for (int node = 0; node < NodeCount; node++)
        {
            stages[node] = (int)rows[3 * node];
            tensorParallel[node] = (int)rows[3 * node + 1];
            capacity[node] = (long)rows[3 * node + 2];
            measured &= capacity[node] > 0;
        }
        if (!measured)
            for (int node = 0; node < NodeCount; node++)
                capacity[node] = tensorParallel[node];
        return new LayerSplitTopology(stages, tensorParallel, capacity);
    }

    public IAllocator GetAllocator(int rank) => LocalGroup.GetAllocator(rank);
    public void Send(float[] buffer, int count, int destination)
    {
        ValidateBuffer(buffer, count);
        _tcp.SendTensor(buffer.AsSpan(0, count), destination);
    }
    public void Receive(float[] buffer, int count, int source)
    {
        ValidateBuffer(buffer, count);
        _tcp.ReceiveTensor(buffer.AsSpan(0, count), source);
    }
    private static void ValidateBuffer(float[] buffer, int count)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if ((uint)count > (uint)buffer.Length) throw new ArgumentOutOfRangeException(nameof(count));
    }
    public void AllReduce(Tensor[] tensors) => throw new NotSupportedException("Whole-layer execution does not reduce weight shards.");
    // Used only to agree on startup metadata, never during layer execution.
    public void CrossNodeAllReduce(float[] buffer, int count)
    {
        ValidateBuffer(buffer, count);
        _tcp.AllReduce(buffer, count);
    }
    public void Synchronize() => LocalGroup.Synchronize();
    public void Barrier() { Synchronize(); _tcp.Barrier(); }
    public void BroadcastControl(int op, int[] payload) => _tcp.BroadcastControl(op, payload);
    public (int op, int[] payload) ReceiveControl() => _tcp.ReceiveControl();
    public void Abort(Exception failure) => _tcp.Abort(failure);
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _tcp.Dispose();
        LocalGroup.Dispose();
    }
}
