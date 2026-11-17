// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// A node of a network program (<see cref="NetworkBuilder"/>) with its extents in ggml's order: <see cref="Ne0"/> is the
    /// innermost (a row of features), up to four. Activations are rows of features: [features, rows]; a 2-D feature map is
    /// [channels, width, height] (channels innermost, the layout token rows already have).
    /// </summary>
    public readonly struct NetValue
    {
        internal NetValue(int id, long ne0, long ne1, long ne2, long ne3, bool strided, int format)
        {
            Id = id;
            Ne0 = ne0;
            Ne1 = ne1;
            Ne2 = ne2;
            Ne3 = ne3;
            Strided = strided;
            Format = format;
            Valid = true;
        }

        /// <summary>The node's index in its program.</summary>
        public int Id { get; }

        public long Ne0 { get; }
        public long Ne1 { get; }
        public long Ne2 { get; }
        public long Ne3 { get; }

        /// <summary>A view whose elements are not packed (a permute, a column block): ops that need packed rows copy it.</summary>
        public bool Strided { get; }

        /// <summary>Element format (<see cref="GraphFormat"/>): F32 activations, I32 indices, F16 masks, a weight's format.</summary>
        public int Format { get; }

        /// <summary>False for default(NetValue): "no value" where a node is optional.</summary>
        public bool Valid { get; }

        public long Elements => Ne0 * Ne1 * Ne2 * Ne3;

        public long[] Ne => new[] { Ne0, Ne1, Ne2, Ne3 };

        public long this[int axis] => axis switch { 0 => Ne0, 1 => Ne1, 2 => Ne2, 3 => Ne3, _ => 1 };

        /// <summary>The number of extents up to the last one above 1 (at least 1).</summary>
        public int Dims => Ne3 > 1 ? 4 : Ne2 > 1 ? 3 : Ne1 > 1 ? 2 : 1;

        public override string ToString() => Valid ? $"#{Id} [{Ne0}, {Ne1}, {Ne2}, {Ne3}]{(Strided ? " strided" : "")}" : "(none)";
    }
}
