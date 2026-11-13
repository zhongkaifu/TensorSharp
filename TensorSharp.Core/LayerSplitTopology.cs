// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;

namespace TensorSharp;

/// <summary>
/// The shape of every node of a multi-node whole-layer pipeline, in node-id order, as the nodes agreed at startup:
/// each node's local stage count (<c>--layer-split</c>), the devices of each of its stages (<c>--tp</c>), and the
/// capacity of one of its stages (the memory of its devices, in whole GiB; any positive unit every node uses alike).
/// Nodes may differ: a node with two GPUs can run one tensor-parallel stage next to a node with one.
/// </summary>
public sealed class LayerSplitTopology
{
    private readonly int[] _stages;
    private readonly int[] _tensorParallel;
    private readonly long[] _stageCapacity;

    public LayerSplitTopology(int[] stages, int[] tensorParallel, long[] stageCapacity)
    {
        ArgumentNullException.ThrowIfNull(stages);
        ArgumentNullException.ThrowIfNull(tensorParallel);
        ArgumentNullException.ThrowIfNull(stageCapacity);
        if (stages.Length < 1 || tensorParallel.Length != stages.Length || stageCapacity.Length != stages.Length)
            throw new ArgumentException("Every node needs a stage count, a tensor-parallel degree and a capacity.");
        for (int node = 0; node < stages.Length; node++)
        {
            if (stages[node] < 1 || tensorParallel[node] < 1)
                throw new ArgumentOutOfRangeException(nameof(stages), $"Node {node} has {stages[node]} stage(s) of {tensorParallel[node]} device(s).");
            if (stageCapacity[node] < 1)
                throw new ArgumentOutOfRangeException(nameof(stageCapacity), $"Node {node} reported no stage capacity.");
        }
        _stages = (int[])stages.Clone();
        _tensorParallel = (int[])tensorParallel.Clone();
        _stageCapacity = (long[])stageCapacity.Clone();
    }

    public int NodeCount => _stages.Length;

    /// <summary>Whole-layer stages node <paramref name="node"/> runs.</summary>
    public int Stages(int node) => _stages[node];

    /// <summary>Devices of each of node <paramref name="node"/>'s stages.</summary>
    public int TensorParallel(int node) => _tensorParallel[node];

    /// <summary>Capacity of one of node <paramref name="node"/>'s stages.</summary>
    public long StageCapacity(int node) => _stageCapacity[node];

    public int TotalStages => _stages.Sum();

    public int TotalDevices => Enumerable.Range(0, NodeCount).Sum(n => _stages[n] * _tensorParallel[n]);

    /// <summary>Devices of the nodes before <paramref name="node"/>.</summary>
    public int DeviceOffset(int node) => Enumerable.Range(0, node).Sum(n => _stages[n] * _tensorParallel[n]);

    /// <summary>Every node has the same stage count and tensor-parallel degree.</summary>
    public bool HasUniformShape => _stages.All(s => s == _stages[0]) && _tensorParallel.All(t => t == _tensorParallel[0]);

    /// <summary>Every node has the same shape and stage capacity: the pipeline splits evenly, as without a topology.</summary>
    public bool IsUniform => HasUniformShape && _stageCapacity.All(c => c == _stageCapacity[0]);

    public string Describe() => string.Join(" | ", Enumerable.Range(0, NodeCount).Select(n =>
        $"node {n}: {_stages[n]} stage(s) x {_tensorParallel[n]} GPU(s), {_stageCapacity[n]} GiB/stage"));
}
