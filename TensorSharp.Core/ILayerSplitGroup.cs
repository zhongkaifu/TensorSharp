using System;

namespace TensorSharp;

/// <summary>
/// Ordered whole-layer pipeline. Each node owns a contiguous range of layers,
/// distributed over its local devices. Activations travel to the next node;
/// the final node returns logits to the driver. No tensor reduction crosses nodes
/// (a node whose stages are tensor-parallel groups reduces within them).
/// </summary>
public interface ILayerSplitGroup : ITensorParallelGroup
{
    int NodeId { get; }

    /// <summary>Devices of each of this node's whole-layer stages: 1 for a pipeline of single devices, T when every
    /// local stage is a tensor-parallel group of T devices (<c>--tp T --layer-split P</c> with <c>--tp-peers</c>).
    /// <see cref="ITensorParallelGroup.Degree"/> counts this node's devices, P x T.</summary>
    int StageTensorParallel => 1;

    /// <summary>This node's whole-layer stages (P).</summary>
    int LocalStages => Degree / StageTensorParallel;

    /// <summary>Every node's shape as the nodes agreed at startup; null when the group does not exchange shapes
    /// (every node is then assumed to have this node's).</summary>
    LayerSplitTopology Topology => null;

    /// <summary>Send exactly <paramref name="count"/> lossless 32-bit words.</summary>
    void Send(float[] buffer, int count, int destination);

    /// <summary>Receive exactly <paramref name="count"/> lossless 32-bit words.</summary>
    void Receive(float[] buffer, int count, int source);
}
