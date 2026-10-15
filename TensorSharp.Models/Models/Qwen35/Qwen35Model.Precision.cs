// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models;

public partial class Qwen35Model
{
    private void RegisterQwenQ8Precision()
    {
        if (_backend != BackendType.GgmlCuda || _numExperts != 0) return;
        // Register after fusing, sharding and preloading, when both host and
        // device-cache identities are stable. Exclude a separately loaded draft.
        var weights = new HashSet<QuantizedWeight>();
        static bool Trunk(string name) => name.StartsWith("blk.", StringComparison.Ordinal)
            || name is "output.weight" or "token_embd.weight";
        foreach (var pair in _quantWeights)
            if (Trunk(pair.Key)) weights.Add(pair.Value);
        foreach (var pair in _tpQuantWeights)
            if (Trunk(pair.Key))
                foreach (var weight in pair.Value) if (weight != null) weights.Add(weight);
        try
        {
            foreach (var weight in weights)
                if (!weight.IsStreamed) weight.EnableQ8F32Activations();
        }
        catch
        {
            // Failed construction must not leave a registered pointer after the
            // model owner is lost. Each weight also unregisters at normal dispose.
            foreach (var weight in weights) weight.DisableQ8F32Activations();
            throw;
        }
    }
}
