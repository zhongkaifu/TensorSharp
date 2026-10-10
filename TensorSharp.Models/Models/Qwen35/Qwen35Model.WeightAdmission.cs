// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

public partial class Qwen35Model
{
    // Direct API callers have historically attached a speculative decoder after
    // construction. Preserve that capability when no policy was supplied. An
    // explicit --no-spec veto, or an enabled n-gram drafter, cannot use NextN.
    // Resolve once: changing process environment later must not change admission.
    internal static bool ShouldLoadEmbeddedMtpWeights(SpeculationOptions options, ModelMemoryPolicy memoryPolicy = null) =>
        memoryPolicy?.OmitEmbeddedDraftWeights != true && !options.ExplicitlyDisabled
        && !(options.Enabled && string.Equals(options.SpeculatorName,
            SpeculatorRegistry.NGram, StringComparison.OrdinalIgnoreCase));

    internal static bool IsEmbeddedMtpWeight(string name, int trunkLayers, int nextnLayers) =>
        nextnLayers > 0 && MoeCpuOffloadConfig.TryParseLayerIndex(name, out int layer)
        && layer >= trunkLayers && (long)layer < (long)trunkLayers + nextnLayers;

    protected override bool ShouldLoadWeight(GgufTensorInfo info) =>
        _loadEmbeddedMtpWeights || !IsEmbeddedMtpWeight(info.Name, Config.NumLayers, _numNextnLayers);
}
