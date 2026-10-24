// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Generic;

namespace TensorSharp.Models.Moe
{
    /// <summary>
    /// One layer's worth of MoE expert weights, stored host-side as flat F32
    /// buffers in expert-major row-major order. A runtime-created expert set is
    /// naturally F32; the integration layer (not yet implemented) is responsible
    /// for quantizing / fusing gate+up / uploading to the backend it needs.
    /// </summary>
    public sealed class ExpertSet
    {
        /// <summary>Number of experts in this set.</summary>
        public int NumExperts { get; init; }

        /// <summary>Expert FFN intermediate dim.</summary>
        public long Nff { get; init; }

        /// <summary>Expert FFN input/attention dim.</summary>
        public long Hidden { get; init; }

        /// <summary><c>[NumExperts, Nff, Hidden]</c> expert-major row-major.</summary>
        public float[] GateWeights { get; init; } = Array.Empty<float>();

        /// <summary><c>[NumExperts, Nff, Hidden]</c> expert-major row-major.</summary>
        public float[] UpWeights { get; init; } = Array.Empty<float>();

        /// <summary><c>[NumExperts, Hidden, Nff]</c> expert-major row-major.</summary>
        public float[] DownWeights { get; init; } = Array.Empty<float>();

        /// <summary>
        /// Optional replacement router projection, <c>[NumExperts, Hidden]</c> row-major.
        /// Null means the base model's router for that layer stays in effect. Set by the
        /// loader during construction; read-only afterward.
        /// </summary>
        public float[]? RouterWeights { get; set; }

        public bool HasRouter => RouterWeights != null;
    }

    /// <summary>
    /// A self-contained MoE expert module targeting one or more layers of a base
    /// model. Immutable after load. <see cref="TargetLayers"/> are, by convention,
    /// the 0-based layer indices the module's experts replace (or extend) in the
    /// running model.
    /// </summary>
    public sealed class MoeModule
    {
        public string Id { get; }

        /// <summary>The base architecture the module was built for (e.g. <c>"gemma4"</c>).</summary>
        public string Architecture { get; }

        /// <summary>Free-form domain tag, e.g. <c>"coding"</c>. May be null.</summary>
        public string? ExpertType { get; }

        /// <summary>Per-layer expert sets keyed by 0-based layer index.</summary>
        public IReadOnlyDictionary<int, ExpertSet> Experts { get; }

        /// <summary>One-line summary of the module for diagnostics / listings.</summary>
        public string Summary { get; }

        public MoeModule(
            string id,
            string architecture,
            string? expertType,
            IReadOnlyDictionary<int, ExpertSet> experts,
            string summary)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("A MoE module id is required.", nameof(id));
            Id = id;
            Architecture = architecture;
            ExpertType = expertType;
            Experts = experts ?? throw new ArgumentNullException(nameof(experts));
            Summary = summary ?? string.Empty;
        }
    }

    /// <summary>Public metadata for administrative listings.</summary>
    public sealed record MoeModuleInfo(
        string Id,
        string Architecture,
        string? ExpertType,
        int LayerCount,
        int NumExperts,
        bool HasRouter);

    /// <summary>
    /// Which module's experts are active for which layers. Layers not present in
    /// the map use the base model's original experts. Router overrides travel with
    /// the module itself (<see cref="ExpertSet.HasRouter"/>); this selection only
    /// decides the expert weights.
    /// </summary>
    public sealed class MoeSelection
    {
        /// <summary>0-based layer index -> module id.</summary>
        public IReadOnlyDictionary<int, string> LayerOverrides { get; init; } =
            new Dictionary<int, string>();
    }
}