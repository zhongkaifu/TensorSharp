// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Globalization;

namespace TensorSharp
{
    /// <summary>
    /// Where a model's weights and work are placed: its tensor-parallel degree, its whole-layer split, the GPUs each uses and
    /// whether it runs across nodes (D04: model-instance scope). A host resolves it once from its input channels - command
    /// line flags first (<c>--tp</c>, <c>--layer-split</c>, <c>--tp-node-id</c>, <c>--tp-peers</c>), then the process
    /// environment (<see cref="FromEnvironment"/>), then the defaults - and hands the validated value to the loader
    /// (<c>ModelBase.Create</c>), which keeps it as the model's snapshot. Modules no longer pass placement to each other by
    /// writing the environment.
    /// </summary>
    public sealed record ModelPlacementOptions
    {
        public const string TpDegreeVariable = "TENSORSHARP_TP_DEGREE";
        public const string LayerSplitDegreeVariable = "TENSORSHARP_LAYER_SPLIT_DEGREE";
        public const string TpDevicesVariable = "TENSORSHARP_TP_DEVICES";
        public const string LayerSplitDevicesVariable = "TENSORSHARP_LAYER_SPLIT_DEVICES";
        public const string NodeIdVariable = "TENSORSHARP_TP_NODE_ID";
        public const string PeersVariable = "TENSORSHARP_TP_PEERS";

        /// <summary>One GPU, one stage, one node.</summary>
        public static ModelPlacementOptions Default { get; } = new();

        /// <summary>Tensor-parallel ranks on this node (1: none).</summary>
        public int TpDegree { get; init; } = 1;

        /// <summary>Whole-layer stages on this node (1: none).</summary>
        public int LayerSplitDegree { get; init; } = 1;

        /// <summary>Whether a layer split was asked for at all (<c>--layer-split</c>, even of 1): across nodes it selects the
        /// node pipeline rather than one tensor-parallel group spanning them.</summary>
        public bool LayerSplitRequested { get; init; }

        /// <summary>The GPUs of the tensor-parallel ranks ("0,2"), or null for 0..degree-1.</summary>
        public string? TpDevices { get; init; }

        /// <summary>The GPUs of the layer-split stages ("1,3"), or null for 0..degree-1.</summary>
        public string? LayerSplitDevices { get; init; }

        /// <summary>This node's id in a multi-node group, or null on one node.</summary>
        public string? DistributedNodeId { get; init; }

        /// <summary>The endpoints of every node of a multi-node group, or null on one node.</summary>
        public string? DistributedPeers { get; init; }

        /// <summary>The placement spans nodes.</summary>
        public bool IsDistributed => !string.IsNullOrWhiteSpace(DistributedNodeId) && !string.IsNullOrWhiteSpace(DistributedPeers);

        /// <summary>Some of the distributed pair is set (a host refuses a load it cannot restart across nodes).</summary>
        public bool NamesDistributedGroup => !string.IsNullOrWhiteSpace(DistributedNodeId) || !string.IsNullOrWhiteSpace(DistributedPeers);

        /// <summary>The placement the process environment describes (the compatibility input channel: what a host without
        /// its own flags, or a direct <c>ModelBase.Create</c> caller, gets). Validated as the flags are.</summary>
        public static ModelPlacementOptions FromEnvironment()
        {
            string? layer = Environment.GetEnvironmentVariable(LayerSplitDegreeVariable);
            return new ModelPlacementOptions
            {
                TpDegree = ReadDegree(Environment.GetEnvironmentVariable(TpDegreeVariable), TpDegreeVariable),
                LayerSplitDegree = ReadDegree(layer, LayerSplitDegreeVariable),
                LayerSplitRequested = !string.IsNullOrWhiteSpace(layer),
                TpDevices = Trimmed(Environment.GetEnvironmentVariable(TpDevicesVariable)),
                LayerSplitDevices = Trimmed(Environment.GetEnvironmentVariable(LayerSplitDevicesVariable)),
                DistributedNodeId = Trimmed(Environment.GetEnvironmentVariable(NodeIdVariable)),
                DistributedPeers = Trimmed(Environment.GetEnvironmentVariable(PeersVariable)),
            };
        }

        /// <summary>The environment's device lists and distributed pair only, without its degrees (a model constructed directly,
        /// with its degrees as arguments, takes the rest of its placement from here).</summary>
        public static ModelPlacementOptions EnvironmentDevices() => new()
        {
            TpDevices = Trimmed(Environment.GetEnvironmentVariable(TpDevicesVariable)),
            LayerSplitDevices = Trimmed(Environment.GetEnvironmentVariable(LayerSplitDevicesVariable)),
            DistributedNodeId = Trimmed(Environment.GetEnvironmentVariable(NodeIdVariable)),
            DistributedPeers = Trimmed(Environment.GetEnvironmentVariable(PeersVariable)),
        };

        /// <summary>The device ordinals of <paramref name="degree"/> ranks (<paramref name="collectives"/>: tensor-parallel
        /// ranks, else layer-split stages): the configured list, or 0..degree-1.</summary>
        public int[] DeviceIds(int degree, bool collectives)
            => ParseDeviceIds(degree, collectives ? TpDevices : LayerSplitDevices, collectives ? TpDevicesVariable : LayerSplitDevicesVariable);

        /// <summary>Parses a device list ("0,2") of exactly <paramref name="degree"/> distinct non-negative ordinals.</summary>
        public static int[] ParseDeviceIds(int degree, string? raw, string name)
        {
            if (string.IsNullOrWhiteSpace(raw))
            {
                var seq = new int[degree];
                for (int i = 0; i < degree; i++) seq[i] = i;
                return seq;
            }
            string[] parts = raw.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != degree)
                throw new ArgumentException($"{name} lists {parts.Length} device(s) but the requested degree is {degree}.");
            var devices = new int[degree];
            for (int i = 0; i < degree; i++)
            {
                if (!int.TryParse(parts[i].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out devices[i]) || devices[i] < 0)
                    throw new ArgumentException($"Invalid device index '{parts[i]}' in {name}.");
                if (Array.IndexOf(devices, devices[i], 0, i) >= 0)
                    throw new ArgumentException($"{name} contains duplicate device index {devices[i]}.");
            }
            return devices;
        }

        private static int ReadDegree(string? raw, string name)
        {
            if (string.IsNullOrWhiteSpace(raw)) return 1;
            if (!int.TryParse(raw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int degree) || degree < 1)
                throw new ArgumentException($"{name} must be a positive integer.");
            return degree;
        }

        private static string? Trimmed(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    }
}
